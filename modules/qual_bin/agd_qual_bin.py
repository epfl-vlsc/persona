from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.python.ops import data_flow_ops, string_ops

from ..common.service import Service
from common.parse import numeric_min_checker, path_exists_checker

import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import queues, pipeline

class QualBinService(Service):
    """ A class representing a service module in Persona """
   
    #default inputs
    def get_shortname(self):
        return "qualbin"

    def output_dtypes(self, args):
        return []
    def output_shapes(self, args):
        return []
    
    def extract_run_args(self, args):
        dataset = args.dataset
        paths = [ a["path"] for a in dataset["records"] ]
        print(paths)
        dataset_dir = args.dataset_dir
        return (os.path.join(dataset_dir, a) for a in paths)

    def add_graph_args(self, parser):
        parser.add_argument("-p", "--parse-parallel", default=1, type=numeric_min_checker(minimum=1, message="read parallelism"),
                            help="total paralellism level for reading data from disk")
        parser.add_argument("-w", "--write-parallel", default=1, help="number of writers to use",
                            type=numeric_min_checker(minimum=1, message="number of writers min"))
        parser.add_argument("-pp", "--process-parallel", default=1, help="adjusting parallelism of quality binning stage",type=numeric_min_checker(minimum=1, message="number of parallel processes"))
        parser.add_argument("-d", "--dataset-dir", type=path_exists_checker(), required=True, help="Directory containing ALL of the chunk files")
	

    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two 
        things: a list of tensors which the runtime will 
        evaluate, and a list of run-once ops"""
        # make the graph
        if not os.path.exists(args.dataset_dir) and os.path.isfile(args.dataset_dir):
            raise EnvironmentError("metadata file '{m}' either doesn't exist or is not a file".format(m=args.dataset))

        if in_queue is None:
            raise EnvironmentError("in queue is none")

        #dataset_dir = os.path.dirname(args.dataset_dir) 
        op = agd_qual_bin_local(in_queue=in_queue,
                                       outdir=args.dataset_dir, 
                                       parallel_parse=args.parse_parallel, 
                                       parallel_write=args.write_parallel, parallel_process = args.process_parallel)
        run_once = []
        return [op], run_once 

def _make_writers(compressed_batch, output_dir, write_parallelism):

    compressed_single = pipeline.join(compressed_batch, parallel=write_parallelism, capacity=8, multi=True)

    for buf, num_recs, first_ordinal, record_id in compressed_single:
    
        first_ord_as_string = string_ops.as_string(first_ordinal, name="first_ord_as_string")
        result_key = string_ops.string_join([output_dir, "/", record_id, "_", first_ord_as_string, ".binned_qual"], name="base_key_string")
        
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

def agd_qual_bin_local(in_queue, outdir=None, parallel_parse=1, parallel_write=1, parallel_compress=1, parallel_process = 2):
    """
    key: tensor with chunk key string
    local_directory: the "base path" from which these should be read
    column_grouping_factor: the number of keys to put together
    parallel_parse: the parallelism for processing records (decomp)
    """
  
    parallel_key_dequeue = tuple(in_queue.dequeue() for _ in range(parallel_parse))
    result_chunks = pipeline.local_read_pipeline(upstream_tensors=parallel_key_dequeue, columns=['qual'])

    result_chunk_list = [ list(c) for c in result_chunks ]

    
    parsed_results = pipeline.agd_reader_multi_column_pipeline(upstream_tensorz=result_chunk_list)
    parsed_results_list = list(parsed_results)

    parsed_result = pipeline.join(parsed_results_list, parallel=parallel_process, capacity=8, multi=True)

    # result_buf, num_recs, first_ord, record_id
    #parsed_results = tf.contrib.persona.persona_in_pipe(key=key, dataset_dir=local_directory, columns=["qual_test"], parse_parallel=parallel_parse,
                                                        #process_parallel=1)
  
    print(parsed_result)
    #result_buf, num_results, first_ord, record_id = parsed_result
    #result_buf = tf.unstack(result_buf)[0]
    #print(result_buf)
	
    bpp = persona_ops.buffer_pair_pool(size=0, bound=False, name="output_buffer_pair_pool")
    
    output_list = []
    for entry in parsed_result:
    	result_buf, num_results, first_ord, record_id = entry
    	print(result_buf)
    	result_buf = tf.unstack(result_buf)[0] 
    	result_out = persona_ops.agd_qual_bin(results_handle=result_buf, num_records=num_results,buffer_pair_pool=bpp, name="qualbinop", upper_bounds = [9, 19, 24, 29, 34, 39,40], bin_values = [6, 15, 22, 27, 33, 37, 40], encoding_offset = 33)
    	output_list.append([result_out, num_results, first_ord, record_id])

    result_to_write = pipeline.join(output_list, parallel=parallel_write, 
        capacity=8, multi=True)
#changed line about

    compressed = compress_pipeline(result_to_write, parallel_compress)

    written = _make_writers(compressed_batch=list(compressed), output_dir=outdir, write_parallelism=parallel_write)
    print(outdir)
    recs = list(written)
    all_written_keys = pipeline.join(recs, parallel=1, capacity=8, multi=False)

    return all_written_keys
  
  
