#!/usr/bin/env python3

import argparse
import os
import subprocess
import functools
import json
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops, string_ops
from tensorflow.python.framework import tensor_shape, dtypes
from tensorflow.python.training import queue_runner

persona_ops = tf.contrib.persona.persona_ops()

def read_pipeline(fastq_files):
    string_producer = tf.train.string_input_producer(fastq_files, num_epochs=1, shuffle=False)
    mapped_file_pool = persona_ops.m_map_pool(size=0, bound=False, name="mmap_pool")
    reader, _ = persona_ops.file_m_map(filename=string_producer.dequeue(), pool_handle=mapped_file_pool, local_prefix="/",
                             synchronous=True, name="file_map")
    unstacked = tf.unstack(reader)
    queued_results = tf.train.batch_pdq(unstacked, batch_size=1, num_dq_ops=1, capacity=2, name="file_map_result_queue")
    return queued_results[0]

def conversion_pipeline(queued_fastq, chunk_size, convert_parallelism):
    q = data_flow_ops.FIFOQueue(capacity=32, # big because who cares
                                dtypes=[dtypes.string, dtypes.int64, dtypes.int64],
                                shapes=[tensor_shape.vector(2), tensor_shape.scalar(), tensor_shape.scalar()],
                                name="chunked_output_queue")
    fastq_read_pool = persona_ops.fastq_read_pool(size=0, bound=False, name="fastq_read_pool")
    chunker = persona_ops.fastq_chunker(chunk_size=chunk_size, queue_handle=q.queue_ref,
                                 fastq_file=queued_fastq, fastq_pool=fastq_read_pool)
    queue_runner.add_queue_runner(queue_runner.QueueRunner(q, [chunker]))
    blp = persona_ops.buffer_list_pool(size=0, bound=False, name="conversion_buffer_pool")
    for _ in range(convert_parallelism):
        fastq_resource, first_ordinal, num_recs = q.dequeue()
        converted = persona_ops.agd_converter(buffer_list_pool=blp, input_data=fastq_resource, name="agd_converter")
        yield converted, first_ordinal, num_recs

record_type = ("base", "qual", "metadata")
def writer_pipeline(converters, write_parallelism, record_id, output_dir):
    prefix_name = tf.Variable("{}_".format(record_id), dtype=dtypes.string, name="prefix_string")
    converted_batch = tf.train.batch_join_pdq(tuple(converters), 1, num_dq_ops=write_parallelism, name="converted_batch_queue")
    for converted_handle, first_ordinal, num_recs in converted_batch:
        first_ord_as_string = string_ops.as_string(first_ordinal, name="first_ord_as_string")
        file_key = string_ops.string_join([prefix_name, first_ord_as_string], name="file_key_string")
        file_path, _ = persona_ops.agd_write_columns(record_id=record_id,
                                                     record_type=record_type,
                                                     column_handle=converted_handle,
                                                     output_dir=output_dir,
                                                     file_path=file_key,
                                                     first_ordinal=first_ordinal,
                                                     num_records=tf.to_int32(num_recs))
        yield file_path, first_ordinal, num_recs

def compressed_writer_pipeline(converters, write_parallelism, record_id, output_dir, compress_parallelism):
    def make_compress():
        to_compress_batch = tf.train.batch_join_pdq(tuple(converters), 1, num_dq_ops=compress_parallelism, name="pre_compression_queue")
        bp = persona_ops.buffer_pool(size=0, bound=False, name="compression_buffer_pool")
        for converted_handle, first_ordinal, num_recs in to_compress_batch:
            yield persona_ops.buffer_list_compressor(buffer_list_size=3, buffer_pool=bp, buffer_list=converted_handle), first_ordinal, num_recs

    prefix_name = tf.Variable("{}_".format(record_id), dtype=dtypes.string, name="prefix_string")
    compressed_batch = tf.train.batch_join_pdq(tuple(make_compress()), 1, num_dq_ops=write_parallelism, name="post_compression_pre_write_queue")

    for compressed_set, first_ordinal, num_recs in compressed_batch:
        first_ord_as_string = string_ops.as_string(first_ordinal, name="first_ord_as_string")
        file_key = string_ops.string_join([prefix_name, first_ord_as_string], name="file_key_string")
        file_path_out = persona_ops.column_writer(record_id=record_id,
                                                  compressed=True,
                                                  record_types=record_type,
                                                  outdir=output_dir,
                                                  columns=compressed_set,
                                                  file_path=file_key,
                                                  first_ordinal=first_ordinal,
                                                  num_recs=tf.to_int32(num_recs))
        yield file_path_out, first_ordinal, num_recs

def get_args():
    def numeric_min(min):
        def check(a):
            a = int(a)
            if a < min:
                raise argparse.ArgumentError("Value must be at least {min}. got {actual}".format(min=min, actual=a))
            return a
        return check

    def make_abs(path, check_func):
        if not (os.path.exists(path) and check_func(path)):
            parser.error("'{}' is not a valid path".format(path))
            return
        return os.path.abspath(path)

    parser = argparse.ArgumentParser(description="Grade-recording script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--chunk", type=numeric_min(1), default=10000, help="chunk size to create records")
    parser.add_argument("-p", "--parallel-conversion", type=numeric_min(1), default=1, help="number of parallel converters")
    parser.add_argument("-n", "--name", required=True, help="name for the record")
    parser.add_argument("-o", "--out", default=".", help="directory to write the final record to")
    parser.add_argument("--logdir", default=".", help="Directory to write tensorflow summary data. Default is PWD")
    parser.add_argument("-w", "--write", default=1, type=numeric_min(1), help="number of parallel writers")
    parser.add_argument("--summary", default=False, action='store_true', help="run with tensorflow summary nodes")
    parser.add_argument("--compress", default=False, action='store_true', help="compress output blocks")
    parser.add_argument("--compress-parallel", default=1, type=numeric_min(1), help="number of parallel compression pipelines")
    parser.add_argument("fastq_files", nargs="+", help="the fastq file to convert")
    args = parser.parse_args()
    if len(args.name) > 31:
        parser.error("Name must be at most 31 characters. Got {}".format(len(args.name)))
    args.fastq_files = [make_abs(p, os.path.isfile) for p in args.fastq_files]
    args.out = os.path.join(make_abs(args.out, os.path.isdir), args.name)
    if os.path.exists(args.out):
        subprocess.run("rm -rf {}".format(args.out), shell=True, check=True)
    os.makedirs(args.out)
    args.logdir = make_abs(args.logdir, os.path.isdir)
    args.compress = False # force false for now
    return args

def make_graph(args):
    reader = read_pipeline(fastq_files=args.fastq_files)
    converters = conversion_pipeline(queued_fastq=reader, chunk_size=args.chunk, convert_parallelism=args.parallel_conversion)
    if args.compress:
        written_records = compressed_writer_pipeline(output_dir=args.out, write_parallelism=args.write, record_id=args.name, converters=converters, compress_parallelism=args.compress_parallel)
    else:
        written_records = writer_pipeline(converters=converters, output_dir=args.out, record_id=args.name, write_parallelism=args.write)
    final_queue = tf.train.batch_join_pdq(tuple(written_records), enqueue_many=False, num_dq_ops=1, batch_size=1, name="written_records_queue")
    return final_queue[0]

def run(args):
    final_tensor = make_graph(args=args)
    ops = final_tensor
    init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    merged = tf.summary.merge_all()
    output_records = []
    output_metadata = {
        "name": args.name, "version": 1, "records": output_records
    }
    with tf.Session() as sess:
        sess.run(init_ops)
        count = 0
        if args.summary:
            summary_writer = tf.summary.FileWriter(logdir=os.path.join(args.logdir, "tf_logdir"), graph=sess.graph)
            ops.append(merged)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        while not coord.should_stop():
            try:
                res = sess.run(ops)
                if args.summary:
                    summary_writer.add_summary(res[-1], global_step=count)
                    count += 1
                    name_bytes, first_ordinal, num_records = res[:-1]
                else:
                    name_bytes, first_ordinal, num_records = res
                first_ordinal = int(first_ordinal)
                num_records = int(num_records)
                output_records.append({
                    'first': first_ordinal,
                    'path': name_bytes.decode(),
                    'last': first_ordinal + num_records
                })
            except tf.errors.OutOfRangeError as oore:
                print("got out of range error: {}".format(oore))
                break
        coord.request_stop()
        coord.join(threads)
    with open(os.path.join(args.out, "{}.json".format(args.name)), 'w+') as f:
        json.dump(output_metadata, f)

if __name__ == "__main__":
    args = get_args()
    run(args=args)
