import argparse
import multiprocessing
import os
import json
import tensorflow as tf
from common.parse import numeric_min_checker, path_exists_checker, non_empty_string_checker
from ..common.service import Service
from ..common import parse
from tensorflow.contrib.persona import queues, pipeline
from tensorflow.python.framework import ops, tensor_shape, common_shapes, constant_op, dtypes
from tensorflow.python.ops import io_ops, variables, string_ops, array_ops, data_flow_ops, math_ops, control_flow_ops

persona_ops = tf.contrib.persona.persona_ops()

class FilteringService(Service):
    """ Filtering """

    #default inputs
    def get_shortname(self):
        return "filtering"
    
    def extract_run_args(self, args):
        dataset_dir = args.dataset_dir
        if dataset_dir is None:
            file_path = args.dataset[parse.filepath_key]
            dataset_dir = os.path.dirname(file_path)
        return (os.path.join(dataset_dir, a) for a in self.extract_run_keys(args=args))

    def extract_run_keys(self, args):
        dataset = args.dataset
        return (a["path"] for a in dataset["records"])

    def output_dtypes(self, args):
        return []
    def output_shapes(self, args):
        return []
    
    def add_graph_args(self, parser):
        # adds the common args to all graphs
        parser.add_argument("-p", "--parallel-parse", type=int, default=1, help="Parallelism of decompress stage")
        parser.add_argument("-o", "--output-path", default="", help="Output dataset directory")
        parser.add_argument("-t", "--threads", type=int, default=multiprocessing.cpu_count()-1, 
          help="Number of threads to use for compression [{}]".format(multiprocessing.cpu_count()-1))
        parser.add_argument("-d", "--dataset-dir", type=path_exists_checker(), help="Directory containing ALL of the chunk files")

    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two 
        things: a list of tensors which the runtime will 
        evaluate, and a list of run-once ops"""
        # make the graph

        ops, run_once = filtering_local(in_queue, args)

        return [ops], run_once

def filtering_local(in_queue, args):
  manifest = args.dataset

  if 'reference' not in manifest:
    raise Exception("No reference data in manifest {}. Unaligned dataset not yet supported. Please align dataset first.".format(args.dataset))

  #bp_handle = persona_ops.buffer_pool(size=10, bound=False, name="buf_pool")
  #mmap_pool = persona_ops.m_map_pool(size=10,  bound=False, name="file_mmap_buffer_pool")
  
  columns = ["base", "qual", "metadata", "results"]
  # num_secondary = 0
  # for column in manifest['columns']:
  #   if 'secondary' in column:
  #     columns.append(column)
  #     secondary += 1

  #print("BAM output using columns: {}".format(columns))
  # TODO  provide option for reading from Ceph

  result_chunks = pipeline.local_read_pipeline(upstream_tensors=[in_queue.dequeue()], columns=columns)

  result_chunk_list = [ list(c) for c in result_chunks ]

  print(result_chunk_list)

  to_parse = pipeline.join(upstream_tensors=result_chunk_list, parallel=args.parallel_parse, multi=True, capacity=8)

  parsed_results = pipeline.agd_reader_multi_column_pipeline(upstream_tensorz=to_parse)

  parsed_results_list = list(parsed_results)

  parsed_result = pipeline.join(parsed_results_list, parallel=1, capacity=8, multi=True)[0]

  # base, qual, meta, result, [secondary], num_recs, first_ord, record_id

  handles = parsed_result[0]
  bases = handles[0]
  quals = handles[1]
  meta = handles[2]
  # give a matrix of all the result columns
  # results = tf.stack(handles[3:])
  results = handles[3]
  num_recs = parsed_result[1]
  first_ord = parsed_result[2]

  q = tf.FIFOQueue(capacity=4, # big because who cares
                               dtypes=[dtypes.int32, dtypes.string, dtypes.string ,dtypes.string, dtypes.string ],
                               shapes=[tensor_shape.scalar(),tensor_shape.vector(2),tensor_shape.vector(2),tensor_shape.vector(2),tensor_shape.vector(2)],
                               name="tensor_queue")

  enqueue_op = q.enqueue([num_recs, results, bases, quals, meta])
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(q, [enqueue_op]))

  op = persona_ops.agd_filtering(tensor_queue=q.queue_ref)

  return [op],[]

  # if args.output_path == "":
  #   output_path = manifest['name'] + ".bam"
  # else:
  #   output_path = args.output_path

  # ref_lens = []
  # ref_seqs = []

  # for contig in manifest['reference_contigs']:
  #   ref_lens.append(contig['length'])
  #   ref_seqs.append(contig['name'])

  # sort = manifest['sort'] if 'sort' in manifest else 'unsorted'

  # pg_id = "personaAGD" # TODO get from manifest
  # read_group = manifest['name'] 
  # agd_to_bam = persona_ops.agd_output_bam(results_handle=results, bases_handle=bases, 
  #                                         qualities_handle=quals, metadata_handle=meta,
  #                                         num_records=num_recs, path=output_path,
  #                                         ref_sequences=ref_seqs, ref_seq_sizes=ref_lens,
  #                                         pg_id=pg_id, read_group=read_group, sort_order=sort,
  #                                         num_threads=args.threads)
  
  # return [agd_to_bam], []

