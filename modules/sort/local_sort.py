#!/usr/bin/env python3
import argparse
import sys
import os
import glob
import json
import time
import datetime

from common import recorder

import tensorflow as tf
from . import agd_merge_sort

def get_args():
    def numeric_min_checker(minimum, message):
        def check_number(n):
            n = int(n)
            if n < minimum:
                raise argparse.ArgumentError("{msg}: got {got}, minimum is {minimum}".format(
                    msg=message, got=n, minimum=minimum
                ))
            return n
        return check_number

    default_dir_help = "Defaults to metadata_file's directory"
    parser = argparse.ArgumentParser(description="Perform a sort on an AGD dataset on a local disk",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--sort-read-parallel", default=1, type=numeric_min_checker(minimum=1, message="read parallelism min for sort phase"),
                        help="total parallelism level for local read pipeline for sort phase")
    parser.add_argument("-c", "--column-grouping", default=5, help="grouping factor for parallel chunk sort",
                        type=numeric_min_checker(minimum=1, message="column grouping min"))
    parser.add_argument("-s", "--sort-parallel", default=1, help="number of sorting pipelines to run in parallel",
                        type=numeric_min_checker(minimum=1, message="sorting pipeline min"))
    parser.add_argument("--sort-process-parallel", default=1, type=numeric_min_checker(minimum=1, message="parallel processing for sort stage"),
                        help="parallel processing pipelines for sorting stage")
    parser.add_argument("-w", "--write-parallel", default=1, help="number of ceph writers to use in parallel",
                        type=numeric_min_checker(minimum=1, message="writing pipeline min"))
    parser.add_argument("--output-name", default="sorted", help="name for the output record")
    parser.add_argument("-b", "--order-by", default="location", choices=["location", "metadata"], help="sort by this parameter [location | metadata]")
    parser.add_argument("--chunk", default=2, type=numeric_min_checker(1, "need non-negative chunk size"), help="chunk size for final merge stage")
    parser.add_argument("--input", help="input directory, containing all the files described in metadata_file\n{}".format(default_dir_help))
    parser.add_argument("--output", help="output directory, where the sorted files should be written\n{}".format(default_dir_help))
    parser.add_argument("--summary", default=False, action='store_true', help="store summary information")
    parser.add_argument("--logdir", default=".", help="Directory to write tensorflow summary data. Default is PWD")
    parser.add_argument("metadata_file", help="the json metadata file describing the chunks in the original result set")

    args = parser.parse_args()
    meta_file = args.metadata_file
    if not os.path.exists(meta_file) and os.path.isfile(meta_file):
        raise EnvironmentError("metadata file '{m}' either doesn't exist or is not a file".format(m=meta_file))

    meta_file_dir = os.path.dirname(meta_file)
    if args.input is None:
        args.input = meta_file_dir
    if args.output is None:
        args.output = meta_file_dir

    metadata = {
        "sort_read_parallelism": args.sort_read_parallel,
        "column_grouping": args.column_grouping,
        "sort_parallelism": args.sort_parallel,
        "write_parallelism": args.write_parallel,
        "sort_process_parallelism": args.sort_process_parallel,
        "order_by": args.order_by
    }

    return args, { "params" : metadata }

def get_metadata_attributes(json_file):
    with open(json_file, 'r') as j:
        metadata = json.load(j)
    records = metadata["records"]
    chunk_keys = list(a["path"] for a in records)
    return chunk_keys

def run_sort(args, metadata):
    g = tf.Graph()
    with g.as_default():
        chunk_keys = get_metadata_attributes(json_file=args.metadata_file)
        all_im_key_op = agd_merge_sort.local_sort_pipeline(file_keys=chunk_keys,
                                                           column_grouping_factor=args.column_grouping,
                                                           local_directory=args.input,
                                                           parallel_read=args.sort_read_parallel,
                                                           parallel_process=args.sort_process_parallel,
                                                           outdir=args.output, parallel_sort=args.sort_parallel,
                                                           order_by=args.order_by)[0]
        init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        merged = tf.summary.merge_all()
    return_keys = []
    with tf.Session(graph=g) as sess:
        ops = [all_im_key_op]
        sess.run(init_ops)

        if args.summary:
            summary_writer = tf.summary.FileWriter("{path}/tf_summary_sort".format(path=args.logdir), graph=sess.graph, max_queue=2**20, flush_secs=10**4)
            count = 0
            ops.append(merged)


        sort_dict = {}
        with recorder.UsageRecorder(sort_dict):
            coord = tf.train.Coordinator()
            print("Starting Sort Run")
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            while not coord.should_stop():
                try:
                    res = sess.run(ops)
                    print("im_path: {im_path}".format(im_path=res[0]))
                    return_keys.append(res[0])
                    if args.summary:
                        summary_writer.add_summary(res[1], global_step=count)
                        count += 1
                except tf.errors.OutOfRangeError as oore:
                    print("got out of range error: {}")#.format(oore))
                    break
            print("Coord requesting stop")
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
        metadata["sort_perf"] = sort_dict
    return return_keys

def clean_keys(intermediate_keys, outdir):
    for fl in ("{outdir}/{fl}".format(outdir=outdir, fl=filename.decode()) for filename in intermediate_keys):
        for fl1 in glob.iglob("{fl}*".format(fl=fl)):
            os.remove(fl1)

def run_merge(args, intermediate_keys, metadata):
    g = tf.Graph()
    with g.as_default():
        final_record_keys = agd_merge_sort.local_merge_pipeline(intermediate_keys=intermediate_keys,
                                                                in_dir=args.output,
                                                                write_parallel=args.write_parallel,
                                                                record_name=args.output_name,
                                                                outdir=args.output,
                                                                chunk_size=args.chunk, order_by=args.order_by)
        init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        merged = tf.summary.merge_all()
    output_records = []
    output_metadata = { 'version': 1,
                        'name': args.output_name,
                        'records': output_records }
    with tf.Session(graph=g) as sess:
        ops = final_record_keys
        sess.run(init_ops)


        if args.summary:
            summary_writer = tf.summary.FileWriter("{path}/tf_summary_merge".format(path=args.logdir), graph=sess.graph, max_queue=2**20, flush_secs=10**4)
            count = 0
            ops.append(merged)
        total_records = 0

        coord = tf.train.Coordinator()

        merge_dict = {}
        with recorder.UsageRecorder(merge_dict):
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            print("Starting Merge Run")
            while not coord.should_stop():
                try:
                    res = sess.run(ops)
                    #import ipdb; ipdb.set_trace()
                    name_bytes, first_ordinal, num_records = res[:-1] if args.summary else res
                    first_ordinal = int(first_ordinal)
                    num_records = int(num_records)
                    total_records += num_records
                    output_records.append({
                        'first': first_ordinal,
                        'path': name_bytes.decode(),
                        'last': first_ordinal + num_records
                    })
                    if args.summary:
                        summary_writer.add_summary(res[-1], global_step=count)
                        count += 1
                except tf.errors.OutOfRangeError as oore:
                    print("got out of range error: {}".format(oore))
                    break
            print("Coord requesting stop")
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
        metadata["num_records"] = total_records
        metadata["merge_perf"] = merge_dict

    clean_keys(intermediate_keys=intermediate_keys, outdir=args.output)
    with open("{directory}/{name}.json".format(directory=args.output, name=args.output_name), 'w') as f:
        json.dump(output_metadata, f)

def run(args, metadata):
    total_dict = {}
    with recorder.UsageRecorder(total_dict):
        im_keys = run_sort(args=args, metadata=metadata)
        final_records = run_merge(args=args, intermediate_keys=im_keys, metadata=metadata)
    metadata["total_perf"] = total_dict

    with open(os.path.join(args.logdir, "metadata_{t}.json".format(t=datetime.datetime.now())), 'w+') as j:
        json.dump(metadata, j)

if __name__ == "__main__":
    args, metadata = get_args()
    run(args=args, metadata=metadata)
