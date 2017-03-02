#!/usr/bin/env python3
from __future__ import print_function

import argparse
import multiprocessing
import os
import shutil

import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()

def setup_output_dir(dirname="cluster_traces"):
    trace_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dirname)
    if os.path.exists(trace_path):
        # nuke it
        shutil.rmtree(trace_path)
    os.makedirs(trace_path)
    return trace_path

def get_args():
    parser = argparse.ArgumentParser(description="Persona instance that runs on a single machine. Usually called via personal.py",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--parallel", type=int, default=2, help="parallel decompression")
    parser.add_argument("-e", "--enqueue", type=int, default=1, help="parallel enqueuing")
    parser.add_argument("-m", "--mmap-queue", type=int, default=2, help="size of the mmaped file record queue")
    parser.add_argument("-a", "--aligners", type=int, default=1, help="number of aligners")
    parser.add_argument("-t", "--aligner-threads", type=int, default=multiprocessing.cpu_count(), help="the number of threads to use per aligner")
    parser.add_argument("-x", "--subchunking", type=int, default=5000, help="the size of each subchunk (in number of reads)")
    parser.add_argument("-w", "--writers", type=int, default=0, help="the number of writer pipelines")
    parser.add_argument("-c", "--compress", default=False, action='store_true', help="compress the output")
    parser.add_argument("-i", "--index-path", default="/scratch/stuart/ref_index")
    parser.add_argument("--paired", default=False, action='store_true', help="interpret dataset as interleaved paired dataset")
    parser.add_argument("--null", type=float, required=False, help="use the null aligner instead of actually aligning")
    parser.add_argument("--deep-verify", default=False, action='store_true', help="verify record integrity")
    parser.add_argument("--local-path", default=None, help="if set, causes the cluster machines to read from this path on their local disks (NOT THE NETWORK)")
    parser.add_argument("--ceph-read-chunk-size", default=(2**26), type=int, help="minimum size to read from ceph storage, in bytes")
    parser.add_argument("--ceph-cluster-name", default="ceph", help="name for the ceph cluster")
    parser.add_argument("--ceph-user-name", default="client.dcsl1024", help="ceph username")
    # TODO this is rigid, needs to be changed to get from the queue service!
    parser.add_argument("--ceph-pool-name", default="chunk_100000", help="ceph pool name")
    parser.add_argument("--ceph-conf-path", default=os.path.join(os.path.dirname(__file__), "ceph_agd_align/ceph.conf"), help="path for the ceph configuration")
    parser.add_argument("--summary", default=False, action="store_true", help="Add TensorFlow summary info to the graph")
    parser.add_argument("--queue-host", help="queue service host")
    parser.add_argument("--upstream-pull-port", type=int, default=5556, help="port to request new work items on")
    parser.add_argument("--downstream-push-port", type=int, default=5557, help="port to send completed work items on")

    args = parser.parse_args()

    ceph_conf_path= args.ceph_conf_path
    if not os.path.exists(ceph_conf_path) and os.path.isfile(ceph_conf_path):
        raise EnvironmentError("Ceph conf path {} either isn't a file, or doesn't exist".format(ceph_conf_path))
    args.ceph_conf_path = os.path.abspath(ceph_conf_path)

    a = args.ceph_read_chunk_size
    if a < 1:
        raise EnvironmentError("Ceph read chunk size most be strictly positive! Got {}".format(a))

    if args.aligners < 1:
        raise EnvironmentError("Must have a strictly positive number of aligners! Got {}".format(args.aligners))

    if args.null is not None:
        if args.null < 0.0:
            raise EnvironmentError("null wait time must be strictly non-negative. Got {}".format(args.null))

    index_path = args.index_path
    if not (os.path.exists(index_path) and os.path.isdir(index_path)):
        raise EnvironmentError("Index path '{}' specified incorrectly. Doesn't exist".format(index_path))

    if args.aligner_threads < args.aligners:
        raise EnvironmentError("Aligner must have at least 1 degree of parallelism! Got {}".format(args.aligner_threads))

    local_path = args.local_path
    if local_path is not None:
        if not (os.path.exists(local_path) and os.path.isdir(local_path)):
            raise EnvironmentError("Local path'{l}' not found".format(l=local_path))

    if args.writers < 0:
        raise EnvironmentError("need a strictly positive number of writers, got {}".format(args.writers))
    if args.parallel < 1:
        raise EnvironmentError("need at least 1 parallel dequeue, got {}".format(args.parallel))
    if args.enqueue < 1:
        raise EnvironmentError("need at least 1 parallel enqueue, got {}".format(args.enqueue))

    return args

def _make_zmq_url(addr, port):
  return "tcp://{addr}:{port}".format(addr=addr, port=port)

def build_queues(server_addr, server_port, parallel_enqueue):
    source_name = persona_ops.zero_mq_source(url=_make_zmq_url(addr=server_addr, port=server_port), name="zmq_source")
    record_queue = tf.train.batch_pdq(tensor_list=[source_name],
                                      batch_size=1, capacity=3, enqueue_many=False,
                                      num_threads=1, num_dq_ops=parallel_enqueue,
                                      name="ready_record_data_queue")
    return record_queue


def create_ops(processed_batch, deep_verify, num_aligners, aligner_threads, subchunk_size, num_writers, index_path, null_align, paired):

    genome = persona_ops.genome_index(genome_location=index_path, name="genome_loader")
    if paired:
        options = persona_ops.paired_aligner_options(cmd_line="-o output.sam", name="paired_aligner_options")
    else:
        options = persona_ops.aligner_options(cmd_line="-o output.sam", name="aligner_options") # -o output.sam will not actually do anything

    pp = persona_ops.buffer_pool(size=1, bound=False)
    drp = persona_ops.agd_read_pool(size=0, bound=False)

    aggregate_enqueue = []
    for chunk in processed_batch:
        key = chunk[0]
        num_reads = chunk[1]
        first_ord = chunk[2]
        base_reads = chunk[3]
        qual_reads = chunk[4]

        agd_read = persona_ops.no_meta_agd_assembler(agd_read_pool=drp,
                                                  base_handle=base_reads,
                                                  qual_handle=qual_reads,
                                                  num_records=num_reads, name="agd_assembler")

        aggregate_enqueue.append((agd_read, key, num_reads, first_ord))

    agd_records = tf.train.batch_join_pdq(tensor_list_list=[e for e in aggregate_enqueue],
                                          batch_size=1, capacity=num_aligners + 2,
                                          enqueue_many=False,
                                          num_dq_ops=num_aligners,
                                          name="agd_reads_to_aligners")
    results = []
    blp = persona_ops.buffer_list_pool(size=1, bound=False)

    extra_aligners = aligner_threads % num_aligners
    threads_per_aligner = aligner_threads // num_aligners

    for i, (record, key, num_records, first_ordinal) in enumerate(agd_records):
        thread_count = threads_per_aligner + 1 if i < extra_aligners else threads_per_aligner
        if null_align is not None:
            aligned_result = persona_ops.null_aligner(buffer_list_pool=blp, read=record,
                                                     subchunk_size=subchunk_size, extra_wait=null_align, name="null_aligner")
        else:
            if paired:
                aligner_type = persona_ops.agd_paired_aligner
                name="snap_paired_aligner"
            else:
                aligner_type = persona_ops.snap_align_agd_parallel
                name="snap_single_aligner"
            aligned_result = aligner_type(genome_handle=genome, options_handle=options, num_threads=thread_count,
                                          buffer_list_pool=blp, read=record, subchunk_size=subchunk_size, name=name)

        result = (aligned_result, key, num_records, first_ordinal)
        results.append(result)

    aligned_results = tf.train.batch_join_pdq(tensor_list_list=[r for r in results],
                                              batch_size=1, capacity=2*num_aligners + 10, #this queue should always be empty!
                                              num_dq_ops=max(1, num_writers), name="results_to_sink")
    return aligned_results, genome, options

def local_write_results(aligned_results, output_path, record_name, compress_output):
    ops = []
    for result_out, key_out, num_records, first_ordinal in aligned_results:
        writer_op = persona_ops.parallel_column_writer(
            column_handle=result_out,
            record_type="results",
            record_id=record_name,
            num_records=num_records,
            first_ordinal=first_ordinal,
            file_path=key_out, name="results_file_writer",
            compress=compress_output, output_dir=output_path
        )
        ops.append(writer_op)
    return ops

def ceph_write_results(aligned_results, record_name, compress_output, cluster_name, user_name, pool_name, ceph_conf_path):
    ops = []
    for result_out, key_out, num_records, first_ordinal in aligned_results:
        num_recs = tf.unstack(num_records, name="num_recs_unstack")
        first_ord = tf.unstack(first_ordinal, name="first_ords_unstack")
        key_passthrough = persona_ops.ceph_writer(
            cluster_name=cluster_name,
            user_name=user_name,
            pool_name=pool_name,
            ceph_conf_path=ceph_conf_path,
            compress=compress_output,
            record_id=record_name,
            record_type="results",
            column_handle=result_out,
            file_name=key_out,
            first_ordinal=first_ord[0],
            num_records=num_recs[0],
            name="Ceph_Writer")
        ops.append(key_passthrough)
    return ops

def run_aligner(sink_op, genomes, summary, null):
    #trace_dir = setup_output_dir()
    init_op = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    ops = [sink_op]


    with tf.Session() as sess:
        if summary:
            trace_dir = setup_output_dir()
            ops.append(merged)
            summary_writer = tf.train.SummaryWriter("{trace_dir}/tf_summary".format(trace_dir=trace_dir), graph=sess.graph, max_queue=2**20, flush_secs=10**4)
            count = 0
        sess.run([init_op])  # required for epoch input variable
        if not null:
            sess.run(genomes)
        coord = tf.train.Coordinator()
        print("Starting Run")
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        while not coord.should_stop():
            try:
                a = sess.run(ops)
                if summary:
                    summary_writer.add_summary(a[-1], global_step=count)
                    count += 1
            except tf.errors.OutOfRangeError:
                print('Got out of range error!')
                break
        print("Coord requesting stop")
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def run(args):
    all_ops = []
    genomes = []

    parallel_enqueue = args.enqueue; parallel_dequeue = args.parallel
    mmap_queue_length = args.mmap_queue; min_ceph_read_size = args.ceph_read_chunk_size
    queue_host = args.queue_host; summary = args.summary
    ceph_pool_name = args.ceph_pool_name; ceph_cluster_name = args.ceph_cluster_name
    ceph_user_name = args.ceph_user_name; ceph_conf_path = args.ceph_conf_path
    local_path = args.local_path; paired = args.paired

    record_batch = build_queues(server_port=args.upstream_pull_port, server_addr=queue_host, parallel_enqueue=parallel_enqueue)

    ceph_params = {}
    ceph_params["user_name"] = ceph_user_name
    ceph_params["ceph_conf_path"] = ceph_conf_path
    ceph_params["cluster_name"] = ceph_cluster_name
    ceph_params["pool_name"] = ceph_pool_name


    pp = persona_ops.buffer_pool(size=1, bound=False)
    if local_path is None:
        parsed_chunks = tf.contrib.persona.persona_ceph_in_pipe(dataset_dir=local_path, columns=["base", "qual"], ceph_params=ceph_params, 
                                    keys=record_batch, buffer_pool=pp, parse_parallel=parallel_dequeue, process_parallel=1)
    else:
        if len(record_batch) > 1:
          raise Exception("Local disk requires read parallelism of 1")
        mmap_pool = persona_ops.m_map_pool(size=10, bound=False, name="file_mmap_buffer_pool")
        parsed_chunks = tf.contrib.persona.persona_in_pipe(dataset_dir=local_path, columns=["base", "qual"], key=record_batch[0],
                                                           mmap_pool=mmap_pool, buffer_pool=pp, parse_parallel=parallel_dequeue, process_parallel=1)

    aligned_results, genome, options = create_ops(processed_batch=parsed_chunks, deep_verify=args.deep_verify,
                                                  num_aligners=args.aligners, num_writers=args.writers, subchunk_size=args.subchunking,
                                                  aligner_threads=args.aligner_threads, index_path=args.index_path, null_align=args.null,
                                                  paired=paired)
    genomes.append(genome)

    if args.writers == 0:
        results_out, key_out, _, _, = aligned_results[0]
        sink_op = persona_ops.buffer_list_sink(results_out)
        ops = [sink_op]
    elif local_path is None:
        # TODO the record name and the pool name are hardcoded for a single run. Need to restart to do different pools!
        ops = ceph_write_results(aligned_results=aligned_results, record_name=ceph_pool_name, compress_output=args.compress,
                                 ceph_conf_path=ceph_conf_path, cluster_name=ceph_cluster_name, pool_name=ceph_pool_name,
                                 user_name=ceph_user_name)
    else:
        ops = local_write_results(aligned_results=aligned_results, output_path=local_path, record_name="bioflow_exp", compress_output=args.compress)

    all_ops.extend(ops)

    done_key = tf.train.batch_join_pdq(tensor_list_list=[(a,) for a in all_ops],
                                       batch_size=1, capacity=100, # arbitrary
                                       enqueue_many=False,
                                       num_dq_ops=1,
                                       name="finished_record_keys")

    sink_op = persona_ops.zero_mq_sink(input=done_key[0], url=_make_zmq_url(addr=queue_host, port=args.downstream_push_port), name="sink_op")

    run_aligner(sink_op=sink_op, genomes=genomes, summary=summary, null=True if args.null else False)

if __name__ == "__main__":
    args = get_args()
    run(args=args)
