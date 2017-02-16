import argparse
import sys
import os
import glob
import json
import time

import tensorflow as tf
from . import agd_mark_duplicates


def get_metadata_attributes(json_file):
    with open(json_file, 'r') as j:
        metadata = json.load(j)
    records = metadata["records"]
    chunk_keys = list(a["path"] for a in records)
    return chunk_keys

def run_mark(args):
    g = tf.Graph()
    with g.as_default():
        chunk_keys = get_metadata_attributes(json_file=args.metadata_file)
        all_im_key_op = agd_mark_duplicates.agd_mark_duplicates_local(file_keys=chunk_keys,
                                                                       local_directory=args.input,
                                                                       outdir=args.input, 
                                                                       parallel_parse=args.parse_parallel, 
                                                                       parallel_write=args.write_parallel)[0]
        init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        merged = tf.summary.merge_all()
    return_keys = []
    with tf.Session(graph=g) as sess:
        ops = [all_im_key_op]
        sess.run(init_ops)

        coord = tf.train.Coordinator()
        print("Starting Mark Duplicates Run")
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        summary_writer = tf.summary.FileWriter("{path}/tf_summary_duplicates".format(path=args.logdir), graph=sess.graph, max_queue=2**20, flush_secs=10**4)
        count = 0
        ops.append(merged)
        start_time = time.time()
        while not coord.should_stop():
            try:
                res = sess.run(ops)
                print("im_path: {im_path}".format(im_path=res[0]))
                return_keys.append(res[0])
                summary_writer.add_summary(res[1], global_step=count)
                count += 1
            except tf.errors.OutOfRangeError as oore:
                print("got out of range error: {}")#.format(oore))
                break
        print("Coord requesting stop")
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        end_time = time.time()
        print("Mark duplicates time: {}".format(end_time - start_time))
    return return_keys


def run(args):
    #print(os.getpid())
    #import ipdb; ipdb.set_trace()
    im_keys = run_mark(args=args)
    print("keys: {}".format(im_keys))

