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

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            if not yes_or_no("WARNING: Directory {} exists. Persona FASTQ import overwrite? ".format(path)):
                sys.exit(0)
            else:
                print("Nuking {} ... ".format(path))
                for the_file in os.listdir(path):
                    file_path = os.path.join(path, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path): 
                            shutil.rmtree(file_path)
                    except Exception as e:
                        raise(e)
        else:
            raise


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
        parser.add_argument("-c", "--chunk", type=numeric_min_checker(1, "chunk size"), default=100000, help="chunk size to create records")
        parser.add_argument("-p", "--parallel-parse", type=int, default=1, help="Parallelism of decompress stage")
        parser.add_argument("-n", "--name", required=True, help="name for the record")
        parser.add_argument("-o", "--out", default=".", help="directory to write the final record to")
        parser.add_argument("-w", "--write", default=1, type=numeric_min_checker(1, "write parallelism"), help="number of parallel writers")
        parser.add_argument("-t", "--threads", type=int, default=multiprocessing.cpu_count()-1, help="Number of threads to use for compression [{}]".format(multiprocessing.cpu_count()-1))
        parser.add_argument("-d", "--dataset-dir", type=path_exists_checker(), help="Directory containing ALL of the chunk files")
        parser.add_argument("--unaligned", default=False, action='store_true', help="Set true if BAM file is unaligned")
        parser.add_argument("--compress-parallel", default=1, type=numeric_min_checker(1, "compress parallelism"), help="number of parallel compression pipelines")
        parser.add_argument("-q", "--query", required=True, type=str, help="The query/predicate according to which dataset has to be filtered")

    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two 
        things: a list of tensors which the runtime will 
        evaluate, and a list of run-once ops"""
        # make the graph

        self.outdir = os.path.abspath(args.out) + '/'
        mkdir_p(self.outdir)
        print("Filter creating new dataset in {}".format(self.outdir))

        self.output_records = []
        columns = ['base', 'qual', 'metadata'] if args.unaligned else ['base', 'qual', 'metadata', 'results']
        self.output_metadata = {
            "name": args.name, "version": 1, "records": self.output_records, "columns": columns
        }

        manifest = args.dataset
        refs = []
        i = 0
        for contig in manifest['reference_contigs']:
            refs.append({'index':i, 'length':contig['length'], 'name':contig['name']})
            i = i + 1

        self.output_metadata['reference_contigs'] = refs
        # self.output_metadata['sort'] = bamfile.header['HD']['SO']

        ops, run_once = filtering_local(self, in_queue, args)

        return ops, run_once

    def on_finish(self, args, results):
        # print("CALLED ON_FINISH\n")
        for res in results:
            # print("in results loop\n")
            if args.unaligned:
                base, qual, meta, first_ordinal, num_records = res[0]
            else:
                base, qual, meta, result, first_ordinal, num_records = res[0]

            first_ordinal = int(first_ordinal)
            num_records = int(num_records)
            name = os.path.basename(os.path.splitext(base.decode())[0])
            self.output_records.append({
                'first': first_ordinal,
                'path': name,
                'last': first_ordinal + num_records
            })
        self.output_records = sorted(self.output_records, key=lambda rec: int(rec['first']))
        # reset with the sorted, i think it makes a copy?
        self.output_metadata['records'] = self.output_records

        with open(self.outdir + args.name + '_metadata.json', 'w+') as f:
            json.dump(self.output_metadata, f, indent=4)

def filtering_local(self, in_queue, args):
  manifest = args.dataset

  if 'reference' not in manifest:
    raise Exception("No reference data in manifest {}. Unaligned dataset not yet supported. Please align dataset first.".format(args.dataset))
  
  columns = ["base", "qual", "metadata", "results"]

  result_chunks = pipeline.local_read_pipeline(upstream_tensors=[in_queue.dequeue()], columns=columns)

  result_chunk_list = [ list(c) for c in result_chunks ]

  print(result_chunk_list)

  to_parse = pipeline.join(upstream_tensors=result_chunk_list, parallel=args.parallel_parse, multi=True, capacity=8)

  parsed_results = pipeline.agd_reader_multi_column_pipeline(upstream_tensorz=to_parse)

  parsed_results_list = list(parsed_results)

  parsed_result = pipeline.join(parsed_results_list, parallel=1, capacity=8, multi=True)[0]

  handles = parsed_result[0]
  bases = handles[0]
  quals = handles[1]
  meta = handles[2]
  results = handles[3]
  num_recs = parsed_result[1]
  first_ord = parsed_result[2]

  q = tf.FIFOQueue(capacity=4, # big because who cares
                               dtypes=[dtypes.int32, dtypes.string, dtypes.string ,dtypes.string, dtypes.string ],
                               shapes=[tensor_shape.scalar(),tensor_shape.vector(2),tensor_shape.vector(2),tensor_shape.vector(2),tensor_shape.vector(2)],
                               name="tensor_queue")

  enqueue_op = q.enqueue([num_recs, results, bases, quals, meta])
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(q, [enqueue_op]))
  bpp = persona_ops.buffer_pair_pool(size=0, bound=False, name="bufpool")

  chunk, num_recs, first_ord = persona_ops.agd_filtering(chunk_size=args.chunk, unaligned=args.unaligned,query=args.query,tensor_queue=q.queue_ref, bufpair_pool=bpp)

  compressors = tuple(compress_pipeline(converters=[[chunk, first_ord, num_recs]], compress_parallelism=args.compress_parallel))

  writers = tuple(writer_pipeline(compressors, args.write, args.name, self.outdir))
  #final = pipeline.join(upstream_tensors=writers, capacity=8, parallel=1, multi=True)[0]

  return writers, []

def compress_pipeline(converters, compress_parallelism):
    converted_batch = pipeline.join(converters, parallel=compress_parallelism, capacity=8, multi=True, name="compress_input")
    buf_pool = persona_ops.buffer_pool(size=0, bound=False, name="bufpool")

    for chunk, first_ord, num_recs in converted_batch:
        cols = tf.unstack(chunk)
        out = []
        for col in cols:
            out.append(persona_ops.buffer_pair_compressor(buffer_pool=buf_pool, buffer_pair=col))

        out_stacked = tf.stack(out)
        yield out_stacked, first_ord, num_recs


def writer_pipeline(compressors, write_parallelism, record_id, output_dir):
    prefix_name = tf.constant("{}_".format(record_id), name="prefix_string")
    compressed_batch = pipeline.join(compressors, parallel=write_parallelism, capacity=8, multi=True, name="write_input")
    types = ['base_compact', 'text', 'text', 'structured']
    exts = ['.base', '.qual', '.metadata', '.results']
    for chunk_stacked, first_ordinal, num_recs in compressed_batch:
        chunks = tf.unstack(chunk_stacked)
        first_ord_as_string = string_ops.as_string(first_ordinal, name="first_ord_as_string")
        paths = []
        for i, chunk in enumerate(chunks):
            key = string_ops.string_join([output_dir, prefix_name, first_ord_as_string, exts[i]], name="key_string")
            paths.append(persona_ops.agd_file_system_buffer_writer(record_id=record_id,
                                                       record_type=types[i],
                                                       resource_handle=chunk,
                                                       path=key,
                                                       compressed=True,
                                                       first_ordinal=first_ordinal,
                                                       num_records=tf.to_int32(num_recs)))
        yield paths + [first_ordinal, num_recs]

