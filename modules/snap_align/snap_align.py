#!/usr/bin/env python3
from __future__ import print_function

import argparse
import getpass
import json
import multiprocessing
import os
import sys
import shutil
import socket
import time
import zmq
from zmq.devices.basedevice import ProcessDevice

import tensorflow as tf

import common.cluster_manager as cluster_manager
import common.lttng_trace as lttng_trace
import common.recorder as recorder

tracepoints = [
    "bioflow:process_key",
    "bioflow:reads_aligned",
    "bioflow:chunk_write",
    "bioflow:chunk_read",
    "bioflow:chunk_aligned",
    "bioflow:ceph_read",
    "bioflow:ceph_write"
]

trace_name = "bioflow_trace"
metadata_name = "run_data.json"

# FIXME for some weird reason, having this in queue_service module causes issues
# idk why
class QueueService:
    def __init__(self, push_port, pull_port):
        queue_device = ProcessDevice(zmq.STREAMER, zmq.PULL, zmq.PUSH)

        queue_device.bind_in("tcp://*:{}".format(push_port))
        queue_device.bind_out("tcp://*:{}".format(pull_port))

        queue_device.setsockopt_in(zmq.IDENTITY, b'PULL')
        queue_device.setsockopt_out(zmq.IDENTITY, b'PUSH')
        self.push_url = "tcp://127.0.0.1:{}".format(push_port)
        self.pull_url = "tcp://127.0.0.1:{}".format(pull_port)
        self.queue_device = queue_device

    def get_pull_socket(self):
        return self._get_socket(sock_type=zmq.PULL, url=self.pull_url)

    def get_push_socket(self):
        return self._get_socket(sock_type=zmq.PUSH, url=self.push_url)

    def _get_socket(self, sock_type, url):
        ctx = zmq.Context()
        sock = ctx.socket(socket_type=sock_type)
        sock.connect(url)
        return sock

    def __enter__(self):
        self.queue_device.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass # TODO maybe shut this down somehow?

def setup_output_dir(dirname="traces"):
    trace_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dirname)
    if os.path.exists(trace_path):
        # nuke it
        shutil.rmtree(trace_path)
    os.makedirs(trace_path)
    return trace_path


def create_meta_dict(hosts, local_path, dataset_params, distribute, parallel_dequeue, parallel_enqueue, mmap_queue, chunk_size, num_aligners, aligner_threads, subchunk_size, num_writers, compress_output, null_align, paired):
    meta_dict = {
        "parallel_agd_process": parallel_dequeue,
        "parallel_read": parallel_enqueue,
        "nodes": distribute,
        "null_align": null_align,
        "num_aligners": num_aligners,
        "num_writers": num_writers,
        "compress_output": compress_output,
        "read_ready_queue_length": mmap_queue,
        "aligner_threads": aligner_threads,
        "aligner_subchunking": subchunk_size,
        "cluster_def": hosts,
        "local_path": local_path if local_path is not None else "",
        "paired": paired
    }
    meta_dict = {
        "params" : meta_dict,
        "chunk_size" : chunk_size
    }
    return meta_dict

def run_computation(send_socket, recv_socket, metadata):
    records = metadata["records"]

    total_records = 0
    for last, first in ((a['last'], a['first']) for a in records):
        total_records += last - first
    print("Total records: {}".format(total_records))

    #input("Press any key to start the run")

    print("Starting run!")
    remaining = set()
    start = time.time()
    for path in (r["path"] for r in records):
        send_socket.send_string(path)
        remaining.add(path)

    remaining_count = len(records)

    while remaining_count > 0:
        msg = recv_socket.recv_string()
        remaining.discard(msg)
        remaining_count -= 1
        if remaining_count % 10 == 0:
            print("{} keys remaining".format(remaining_count))

    if len(remaining) > 0:
        print("Done, but didn't get correct ack for keys: {}".format(remaining))

    end = time.time()
    total_time = end - start
    print("total time: %s" % (total_time))
    metadata["start_time"] = start
    metadata["end_time"] = end
    return total_time, total_records

def run_aligner(hosts, meta_dict, remote_prep_path, local_tensorflow_path, params, queue_host, metadata):
    trace_dir = setup_output_dir()
    lttng_out_dir = "{td}/lttng_trace".format(td=trace_dir)

    usage_recorder = recorder.RemoteRecorder()
    print(hosts)
    network_recorder = recorder.NetworkRecorder(machines=hosts)
    with QueueService(push_port=5555, pull_port=5556) as input_service, QueueService(push_port=5557, pull_port=5558) as output_service, usage_recorder:
        source_socket = input_service.get_push_socket()
        sink_socket = output_service.get_pull_socket()
        with cluster_manager.ClusterManager(hosts=hosts, remote_prep_path=remote_prep_path, wait_time=30,
                                            local_tensorflow_path=local_tensorflow_path, param_string=params, queue_host=queue_host) as pid_map:
            usage_recorder.record_pids(machines_and_pids=pid_map)
            with lttng_trace.LTTngRemoteTracer(remote_addrs=hosts, trace_events=tracepoints, trace_output_dir=lttng_out_dir), network_recorder:
                runtime, num_records = run_computation(send_socket=source_socket, recv_socket=sink_socket, metadata=metadata)

    meta_dict["recorded_stats"] = usage_recorder.get_result_map()
    meta_dict["runtime"] = runtime
    meta_dict["network_usage"] = network_recorder.results
    meta_dict["alignments"] = num_records
    alignments_per_sec = num_records / float(runtime)
    meta_dict["alignments_per_sec"] = alignments_per_sec
    print("Aligments / sec average: {rate}".format(rate=alignments_per_sec))

    with open("{td}/{meta_name}".format(td=trace_dir, meta_name=metadata_name), 'w') as metafile:
        metafile.write(json.dumps(meta_dict))

def run(args):
    distribute = args.distribute; hosts = args.hosts; unload_master = args.unload_master
    local_path = args.local_path; summary = args.summary; nulltime = args.null
    index_path= args.index_path; paired = args.paired

    with open(args.json_file, 'r') as j:
        dataset_params = json.load(j)

    if "pool" in dataset_params:
        ceph_pool = dataset_params["pool"]
    elif local_path is None:
        raise Exception("Must use a ceph-style dataset description. Input file '{}' doesn't have the 'pool' key".format(args.json_file))
    else:
        ceph_pool=""

    records = dataset_params["records"]
    # TODO this should probably be done in tensorflow itself
    if paired:
        for record in records:
            chunk_size = record["last"] - record["first"]
            if chunk_size % 2 != 0:
                raise Exception("Got odd chunk size with paired:\n{record}".format(record=record))

    first_record = records[0]
    chunk_size = first_record["last"] - first_record["first"]

    # TODO this needs to be adjusted, based on other queue sizes
    queue_depth = 5

    hostname = socket.gethostname()
    all_hosts = [a for a in hosts if not unload_master or hostname not in a]
    effective_cluster_size = len(all_hosts)
    if effective_cluster_size < distribute:
        raise Exception("Requested {req} cluster machines, but cluster spec only has {actual}".format(req=distribute,
                                                                                                      actual=effective_cluster_size))
    hosts = all_hosts[:distribute]

    meta_dict = create_meta_dict(parallel_dequeue=args.parallel, hosts=hosts, dataset_params=dataset_params,
                                 parallel_enqueue=args.enqueue, mmap_queue=queue_depth, chunk_size=chunk_size,
                                 num_aligners=args.aligners, aligner_threads=args.aligner_threads,
                                 distribute=args.distribute, subchunk_size=args.subchunking,
                                 local_path=local_path, num_writers=args.writers, compress_output=args.compress, null_align=nulltime,
                                 paired=args.paired)

    params = "-p {parallel} --index {index_path} -e {enqueue} -m {mmapq} -a {aligners} -t {threads} -x {subchunk}" \
             " -w {writers} {paired} {ceph_pool} {nulltime} {compress} {summary}" \
             " {local_path}".format(parallel=args.parallel,
                                    enqueue=args.enqueue, mmapq=queue_depth, aligners=args.aligners, threads=args.aligner_threads, subchunk=args.subchunking,
                                    writers=args.writers, compress="-c" if args.compress else "", summary="--summary" if summary else "",
                                    nulltime="--null {n}".format(n=nulltime) if nulltime is not None else "",
                                    ceph_pool="--ceph-pool-name {c}".format(c=ceph_pool) if local_path is None else "",
                                    local_path="--local-path {l}".format(l=local_path) if local_path is not None else "",
                                    paired="--paired" if paired else "",
                                    index_path=index_path)

    run_aligner(hosts=hosts, meta_dict=meta_dict,
                remote_prep_path=args.remote_path,
                local_tensorflow_path=args.local_tensorflow,
                params=params,
                queue_host=hostname,
                metadata=dataset_params)

