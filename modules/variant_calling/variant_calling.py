
import os
import json
import argparse
import multiprocessing
import tensorflow as tf
from common.parse import numeric_min_checker, path_exists_checker, non_empty_string_checker
from ..common.service import Service
from ..common import parse
from tensorflow.contrib.persona import queues, pipeline
from tensorflow.python.ops import io_ops, variables, string_ops, array_ops, data_flow_ops, math_ops, control_flow_ops
from tensorflow.python.framework import ops, tensor_shape, common_shapes, constant_op, dtypes

persona_ops = tf.contrib.persona.persona_ops()

class PrintChunkService(Service):
    """ BAM Export Service """

    #default inputs
    def get_shortname(self):
        return "variant_call"

    def add_run_args(self,parser):
        parse.add_multi_dataset(parser)

    # def extract_run_args(self, args):
    #     bigger_list = []
    #     for i in range(1,len(args.dataset_dir)):
    #         bigger_list.append(self.extract_run_args_small(args=args,i=i))
    #     return bigger_list


    def extract_run_args(self, args):
        list_all = []
        for i in range(len(args.dataset)):
            sddd = []

            file_path = args.dataset[i][parse.filepath_key]
            dataset_dir = os.path.dirname(file_path)
            for a in self.extract_run_keys(args=args,i=i):
                sddd.append(os.path.join(dataset_dir, a))
            list_all.append(sddd)
        return list_all

    def extract_run_keys(self, args,i):
        dataset = args.dataset[i]
        return (a["path"] for a in dataset["records"])


    # def extract_run_args(self,args):
    #     return (((os.path.join(args.dataset_dir[i], a) for a in self.extract_run_keys(args=args,i=i))) for i in range(0,len(args.dataset_dir)))
    #
    # def extract_run_keys(self, args,i):
    #     dataset = args.dataset[i]
    #     return (a["path"] for a in dataset["records"])



    def output_dtypes(self, args):
        return []
    def output_shapes(self, args):
        return []

    def add_graph_args(self, parser):
        # adds the common args to all graphs
        parser.add_argument("-p", "--parallel-parse", type=int, default=1, help="Parallelism of decompress stage")
        parser.add_argument("-d", "--dataset-dir",nargs = '+' ,type=path_exists_checker(), help="Directory containing ALL of the chunk files")
        parser.add_argument("--free",type=str ,default = "",help="Arguments for freebayes")
        # parser.add_argument("-i", "--dataset-dir", type=path_exists_checker(), required=True, help="Directory containing ALL of the chunk files")
    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two
        things: a list of tensors which the runtime will
        evaluate, and a list of run-once ops"""
        # make the graph

        ops, run_once = print_local(in_queue, args)

        return [ops], run_once

def print_local(in_queue, args):
  q = []
  for i in range(len(in_queue)):
      columns = ["base", "qual", "metadata", "results"]

      result_chunks = pipeline.local_read_pipeline(upstream_tensors=[in_queue[i].dequeue()], columns=columns)

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
      results = handles[3]
      num_recs = parsed_result[1]
      first_ord = parsed_result[2]
      q.append(tf.FIFOQueue(capacity=4, # big because who cares
                                   dtypes=[dtypes.string, dtypes.string ,dtypes.string,dtypes.string, dtypes.int32],
                                   shapes=[tensor_shape.vector(2),tensor_shape.vector(2),tensor_shape.vector(2),tensor_shape.vector(2),tensor_shape.scalar()],
                                   name="tensor_input_queue"))

      enqueue_op = q[i].enqueue([bases, quals, meta , results, num_recs])
      tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(q[i], [enqueue_op]))

  ref_vector = []
  for i in range (len(q)):
    ref_vector.append(q[i].queue_ref)

  manifest = args.dataset[0] # assuming that one of the file can provide the referenceSequence
  if 'reference' not in manifest:
      raise Exception("No reference data in manifest {}. Unaligned BAM not yet supported. Please align dataset first.".format(args.dataset))

  ref_lens = []
  ref_seqs = []
  for contig in manifest['reference_contigs']:
     ref_lens.append(contig['length'])
     ref_seqs.append(contig['name'])

  result = persona_ops.variant_calling(ref_sequences=ref_seqs,ref_seq_sizes=ref_lens,cmd_line = args.free.split(" "),num_datasets=len(q),tensor_queue=ref_vector)

  return [result], []
