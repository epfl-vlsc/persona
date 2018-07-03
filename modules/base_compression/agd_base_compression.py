from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from tensorflow.python.ops import data_flow_ops, string_ops

from ..common.service import Service
from common.parse import numeric_min_checker, path_exists_checker
from ..common import parse

import tensorflow as tf
import ipdb;

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import queues, pipeline

class AGDBaseCompressionService(Service):
    """ A class representing a service module in Persona """

    def get_shortname(self):
        return "agdbcomp"

    def output_dtypes(self, args):
        return []
    def output_shapes(self, args):
        return []

    def extract_run_args(self, args):
        dataset = args.dataset
        paths = [ a["path"] for a in dataset["records"] ]
        #print(paths)
        dataset_dir = args.dataset_dir
        return (os.path.join(dataset_dir, a) for a in paths)

    def add_graph_args(self, parser):
        parser.add_argument("-p", "--parse-parallel", default=1, type=numeric_min_checker(minimum=1, message="read parallelism"),
                            help="total paralellism level for reading data from disk")
        parser.add_argument("-w", "--write-parallel", default=1, help="number of writers to use",
                            type=numeric_min_checker(minimum=1, message="number of writers min"))
        parser.add_argument("-d", "--dataset-dir", type=path_exists_checker(), required=True, help="Directory containing ALL of the chunk files")
        parser.add_argument("-c", "--compress-parallel", default=1, help="number of compressor",
                            type=numeric_min_checker(minimum=1, message="number of compressor min"))

    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two
        things: a list of tensors which the runtime will
        evaluate, and a list of run-once ops"""
        # make the graph
        if not os.path.exists(args.dataset_dir) and os.path.isfile(args.dataset_dir):
            raise EnvironmentError("metadata file '{m}' either doesn't exist or is not a file".format(m=args.dataset))

        if in_queue is None:
            raise EnvironmentError("in queue is none")

        # print("args qui est obtenue ")
        # print(args.dataset_dir)
        dataset_dir = args.dataset_dir
        # dataset_dir = os.path.dirname(args.dataset_dir)
        # print("dataset_dir = :")
        # print(dataset_dir)
        # op = agd_base_compression_local(in_queue=in_queue,
        #                                parallel_parse=args.parse_parallel)
        op = agd_base_compression_local(in_queue=in_queue,
                                       outdir=dataset_dir,
                                       parallel_parse=args.parse_parallel,
                                       parallel_write=args.write_parallel,
                                       parallel_compress = args.compress_parallel)

        columns = args.dataset['columns']
        if "refcompress" not in columns:
            columns.append('refcompress ')
            args.dataset['columns'] = columns
        with open(args.dataset[parse.filepath_key], 'w+') as f:
            args.dataset.pop(parse.filepath_key,None)
            json.dump(args.dataset, f, indent=4)

        run_once = []
        return [[op]], run_once

def _make_writers(compressed_batch, output_dir, write_parallelism):

    compressed_single = pipeline.join(compressed_batch, parallel=write_parallelism, capacity=8, multi=True, name="make_writers_join")

    for buf, num_recs, first_ordinal, record_id in compressed_single:

        first_ord_as_string = string_ops.as_string(first_ordinal, name="first_ord_as_string")

        result_key = string_ops.string_join([output_dir, "/", record_id, "_", first_ord_as_string, ".refcompress"], name="base_key_string")
        result = persona_ops.agd_file_system_buffer_writer(record_id=record_id,
                                                     record_type="text",
                                                     resource_handle=buf,
                                                     path=result_key,
                                                     compressed=True,
                                                     first_ordinal=first_ordinal,
                                                     num_records=tf.to_int32(num_recs))
        yield result # writes out the file path key (full path)

def compress_pipeline(results, compress_parallelism):

    buf_pool = persona_ops.buffer_pool(size=0, bound=False, name="bufpool")

    for buf, num_recs, first_ord, record_id in results:
        compressed_buf = persona_ops.buffer_pair_compressor(buffer_pool=buf_pool, buffer_pair=buf)

        yield compressed_buf, num_recs, first_ord, record_id


# def agd_base_compression_local(in_queue, outdir=None, parallel_parse=1, parallel_write=1, parallel_compress=1):
def agd_base_compression_local(in_queue,outdir=None,parallel_parse=1,parallel_write=1,parallel_compress=1):
    """
    key: tensor with chunk key string
    local_directory: the "base path" from which these should be read
    column_grouping_factor: the number of keys to put together
    parallel_parse: the parallelism for processing records (decomp)
    """

    parallel_key_dequeue = tuple(in_queue.dequeue() for _ in range(parallel_parse))
    result_chunks = pipeline.local_read_pipeline(upstream_tensors=parallel_key_dequeue, columns=['results','base'])

    result_chunk_list = [ list(c) for c in result_chunks ]


    parsed_results = pipeline.agd_reader_multi_column_pipeline(upstream_tensorz=result_chunk_list)
    parsed_results_list = list(parsed_results)

    para = 2
    parsed_result = pipeline.join(parsed_results_list, parallel=para, capacity=8, multi=True, name="parsed_result_join")

    result_out = [None]*para
    result_buf = [None]*para
    num_results = [None]*para
    first_ord = [None]*para
    record_id = [None]*para
    res = [None]*para

    for i in range(para):
    # result_buf, num_recs, first_ord, record_id
    #parsed_results = tf.contrib.persona.persona_in_pipe(key=key, dataset_dir=local_directory, columns=["results"], parse_parallel=parallel_parse,
                                                        #process_parallel=1)
        result_buf[i], num_results[i], first_ord[i], record_id[i] = parsed_result[i]
        result_buf_bis,record_buf_bis = tf.unstack(result_buf[i])

        bpp = persona_ops.buffer_pair_pool(size=0, bound= False, name="output_buffer_pair_pool")
        result_out[i] = persona_ops.agd_base_compression(unpack=True,results=result_buf_bis,
            records=record_buf_bis,chunk_size=num_results[i],buffer_pair_pool=bpp)

    for i in range(para):
        res[i] = [result_out[i], num_results[i], first_ord[i], record_id[i]]

    result_to_write = pipeline.join(res, parallel=parallel_write,
        capacity=8, multi=True, name="result_to_write_join")

    compressed = compress_pipeline(result_to_write, parallel_compress)
    written = _make_writers(compressed_batch=list(compressed), output_dir=outdir, write_parallelism=parallel_write)
    recs = list(written)
    all_written_keys = pipeline.join(recs, parallel=1, capacity=8, multi=False, name="all_written_keys_join")

    return all_written_keys
