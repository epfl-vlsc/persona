
import tensorflow as tf
import argparse
import os
import errno
import json
from ..common.service import Service
from ..common.parse import numeric_min_checker
from tensorflow.python.ops import data_flow_ops, string_ops
from tensorflow.python.framework import tensor_shape, dtypes
from tensorflow.python.training import queue_runner

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import queues, pipeline

def yes_or_no(question):
    # could this overflow the stack if the user was very persistent?
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Please enter ")

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

class ImportFastqService(Service):
   
    #default inputs
    def get_shortname(self):
        return "import_fastq"

    def add_graph_args(self, parser):

        parser.add_argument("-c", "--chunk", type=numeric_min_checker(1, "chunk size"), default=100000, help="chunk size to create records")
        parser.add_argument("-p", "--parallel-conversion", type=numeric_min_checker(1, "parallel conversion"), default=1, help="number of parallel converters")
        parser.add_argument("-n", "--name", required=True, help="name for the record")
        parser.add_argument("-o", "--out", default=".", help="directory to write the final record to")
        parser.add_argument("-w", "--write", default=1, type=numeric_min_checker(1, "write parallelism"), help="number of parallel writers")
        parser.add_argument("--summary", default=False, action='store_true', help="run with tensorflow summary nodes")
        parser.add_argument("--paired", default=False, action='store_true', help="interpret fastq files as paired, requires an even number of files for positional args fastq_files")
        parser.add_argument("--compress", default=False, action='store_true', help="compress output blocks")
        parser.add_argument("--compress-parallel", default=1, type=numeric_min_checker(1, "compress parallelism"), help="number of parallel compression pipelines")
        parser.add_argument("fastq_files", nargs="+", help="the fastq file to convert")

    def add_run_args(self, parser):
        pass

    def distributed_capability(self):
        return False

    def output_dtypes(self):
        return []

    def output_shapes(self):
        return []

    def extract_run_args(self, args):
        # the fastq file names
        if args.paired and not (len(args.fastq_files) % 2 == 0):
            raise Exception("Paired conversion requires even number of fastq files.")
        print(args.fastq_files)
        return args.fastq_files

    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two 
        things: a list of tensors which the runtime will 
        evaluate, and a list of run-once ops"""
        # make the graph

        self.outdir = os.path.abspath(args.out) + '/'
        mkdir_p(self.outdir)
        print("FASTQ import creating new dataset in {}".format(self.outdir))

        input_tensor = in_queue.dequeue()
        #reshaped = tf.reshape(input_tensor, tf.TensorShape([1]))
        if args.paired:
            files = tf.train.batch([input_tensor], batch_size=2)
        else:
            files = input_tensor

        self.output_records = []
        self.output_metadata = {
            "name": args.name, "version": 1, "records": self.output_records, "columns": ['base', 'qual', 'metadata']
        }

        reader = read_pipeline(fastq_file=files, args=args)
        converters = list(conversion_pipeline(queued_fastq=reader, chunk_size=args.chunk, 
                convert_parallelism=args.parallel_conversion, args=args))


        writers = list(writer_pipeline(converters, args.write, args.name, self.outdir))
        final = pipeline.join(upstream_tensors=writers, capacity=8, parallel=1, multi=True)[0]

        return final, []
    
    def on_finish(self, args, results):
        for res in results:
            base, qual, meta, first_ordinal, num_records = res
            first_ordinal = int(first_ordinal)
            num_records = int(num_records)
            name = os.path.basename(os.path.splitext(base.decode())[0])
            self.output_records.append({
                'first': first_ordinal,
                'path': name,
                'last': first_ordinal + num_records
            })
        with open(self.outdir + args.name + '.metadata', 'w+') as f:
            json.dump(self.output_metadata, f, indent=4)

def read_pipeline(fastq_file, args):
    mapped_file_pool = persona_ops.m_map_pool(size=0, bound=False, name="mmap_pool")
    if args.paired:
        assert(fastq_file.get_shape() == tensor_shape.vector(2))
        files = tf.unstack(fastq_file) 
        reader_0 = persona_ops.file_m_map(filename=files[0], pool_handle=mapped_file_pool, 
                                 synchronous=False, name="file_map_0")
        reader_1 = persona_ops.file_m_map(filename=files[1], pool_handle=mapped_file_pool,
                                 synchronous=False, name="file_map_1")
        queued_results = pipeline.join([reader_0, reader_1], parallel=1, capacity=2)
    else:
        reader = persona_ops.file_m_map(filename=fastq_file, pool_handle=mapped_file_pool,
                                 synchronous=False, name="file_map")
        queued_results = pipeline.join(reader, parallel=1, capacity=2)
    return queued_results[0]

def writer_pipeline(converters, write_parallelism, record_id, output_dir):
    prefix_name = tf.Variable("{}_".format(record_id), dtype=dtypes.string, name="prefix_string")
    converted_batch = pipeline.join(converters, parallel=write_parallelism, capacity=8, multi=True)

    for base, qual, meta, first_ordinal, num_recs in converted_batch:
        first_ord_as_string = string_ops.as_string(first_ordinal, name="first_ord_as_string")
        base_key = string_ops.string_join([output_dir, prefix_name, first_ord_as_string, ".base"], name="base_key_string")
        qual_key = string_ops.string_join([output_dir, prefix_name, first_ord_as_string, ".qual"], name="qual_key_string")
        meta_key = string_ops.string_join([output_dir, prefix_name, first_ord_as_string, ".metadata"], name="metadata_key_string")
        base_path = persona_ops.agd_file_system_buffer_pair_writer(record_id=record_id,
                                                     record_type="raw",
                                                     resource_handle=base,
                                                     path=base_key,
                                                     first_ordinal=first_ordinal,
                                                     num_records=tf.to_int32(num_recs))
        qual_path = persona_ops.agd_file_system_buffer_pair_writer(record_id=record_id,
                                                     record_type="raw",
                                                     resource_handle=qual,
                                                     path=qual_key,
                                                     first_ordinal=first_ordinal,
                                                     num_records=tf.to_int32(num_recs))
        meta_path = persona_ops.agd_file_system_buffer_pair_writer(record_id=record_id,
                                                     record_type="raw",
                                                     resource_handle=meta,
                                                     path=meta_key,
                                                     first_ordinal=first_ordinal,
                                                     num_records=tf.to_int32(num_recs))
        yield base_path, qual_path, meta_path, first_ordinal, num_recs

def conversion_pipeline(queued_fastq, chunk_size, convert_parallelism, args):
    if not args.paired:
        q = data_flow_ops.FIFOQueue(capacity=32, # big because who cares
                                dtypes=[dtypes.string, dtypes.int64, dtypes.int64],
                                shapes=[tensor_shape.vector(2), tensor_shape.scalar(), tensor_shape.scalar()],
                                name="chunked_output_queue")
    else:
        q = data_flow_ops.FIFOQueue(capacity=32, # big because who cares
                                dtypes=[dtypes.string, dtypes.string, dtypes.int64, dtypes.int64],
                                shapes=[tensor_shape.vector(2), tensor_shape.vector(2), tensor_shape.scalar(), tensor_shape.scalar()],
                                name="chunked_output_queue")

    fastq_read_pool = persona_ops.fastq_read_pool(size=0, bound=False, name="fastq_read_pool")

    if args.paired:
        chunker = persona_ops.fastq_interleaved_chunker(chunk_size=chunk_size, queue_handle=q.queue_ref,
                                     fastq_file_0=queued_fastq[0], fastq_file_1=queued_fastq[1], 
                                     fastq_pool=fastq_read_pool)
    else:
        chunker = persona_ops.fastq_chunker(chunk_size=chunk_size, queue_handle=q.queue_ref,
                                     fastq_file=queued_fastq, fastq_pool=fastq_read_pool)

    queue_runner.add_queue_runner(queue_runner.QueueRunner(q, [chunker]))
    bpp = persona_ops.buffer_pair_pool(size=0, bound=False, name="conversion_buffer_pool")
    for _ in range(convert_parallelism):
        if args.paired:
            fastq_resource_0, fastq_resource_1, first_ordinal, num_recs = q.dequeue()
            base, qual, meta = persona_ops.agd_interleaved_converter(buffer_pair_pool=bpp, input_data_0=fastq_resource_0,
                    input_data_1=fastq_resource_1, name="agd_converter")
            yield base, qual, meta, first_ordinal, num_recs
        else:
            fastq_resource, first_ordinal, num_recs = q.dequeue()
            base, qual, meta = persona_ops.agd_converter(buffer_pair_pool=bpp, input_data=fastq_resource,
                    name="agd_converter")
            yield base, qual, meta, first_ordinal, num_recs



