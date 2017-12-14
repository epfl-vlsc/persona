
import tensorflow as tf
import shutil
import sys
import os
import errno
import json
from ..common.service import Service
from common.parse import numeric_min_checker, yes_or_no
from tensorflow.python.ops import data_flow_ops, string_ops
from tensorflow.python.framework import tensor_shape, dtypes
from tensorflow.python.training import queue_runner

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import queues, pipeline

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            if not yes_or_no("WARNING: Directory {} exists. Persona FASTA import overwrite? ".format(path)):
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
            raise(exc)

class ImportFastaService(Service):
  
    #default inputs
    def get_shortname(self):
        return "import_fasta"

    def add_graph_args(self, parser):

        # TODO sane defaults depending on num schedulable cores
        parser.add_argument("-c", "--chunk", type=numeric_min_checker(1, "chunk size"), default=100000, help="chunk size to create records")
        parser.add_argument("--dna", action='store_true', help="Set if the input fasta is DNA nucleotides")
        parser.add_argument("--protein", action='store_true', help="Set if the input fasta is protein amino acids")
        parser.add_argument("-p", "--parallel-conversion", type=numeric_min_checker(1, "parallel conversion"), default=1, help="number of parallel converters")
        parser.add_argument("-n", "--name", required=True, help="name for the record")
        parser.add_argument("-o", "--out", default=".", help="directory to write the final record to")
        parser.add_argument("-w", "--write", default=2, type=numeric_min_checker(1, "write parallelism"), help="number of parallel writers")
        parser.add_argument("--compress-parallel", default=10, type=numeric_min_checker(1, "compress parallelism"), help="number of parallel compression pipelines")
        parser.add_argument("fasta_file", help="the fasta file to convert")

    def add_run_args(self, parser):
        pass

    def distributed_capability(self):
        return False

    def output_dtypes(self, args):
        return []

    def output_shapes(self, args):
        return []

    def extract_run_args(self, args):
        # the fasta file names
        print(args.fasta_file)
        return [args.fasta_file]

    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two 
        things: a list of tensors which the runtime will 
        evaluate, and a list of run-once ops"""
        # make the graph

        if args.dna and args.protein:
            raise Exception("FASTA files cannot contain both protein and DNA/RNA sequences.")
        if args.protein:
            self.suffix = "prot"
            args.dna = False;
        else:
            self.suffix = "base"

        self.outdir = os.path.abspath(args.out) + '/'
        mkdir_p(self.outdir)
        print("FASTA import creating new dataset in {}".format(self.outdir))

        input_tensor = in_queue.dequeue()
        #reshaped = tf.reshape(input_tensor, tf.TensorShape([1]))
        files = input_tensor

        self.output_records = []
        self.output_metadata = {
            "name": args.name, "version": 1, "records": self.output_records, "columns": [self.suffix, 'metadata']
        }

        reader = read_pipeline(fasta_file=files, args=args)
        converters = tuple(conversion_pipeline(queued_fasta=reader, chunk_size=args.chunk,
                                               convert_parallelism=args.parallel_conversion, args=args))

        compressors = tuple(compress_pipeline(converters=converters, compress_parallelism=args.compress_parallel))

        writers = tuple(writer_pipeline(compressors, args.write, args.name, self.outdir, self.suffix, args))
        #final = pipeline.join(upstream_tensors=writers, capacity=8, parallel=1, multi=True)[0]

        return writers, []
    
    def on_finish(self, args, results):
        for res in results:
            base, meta, first_ordinal, num_records = res[0]
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
        self.output_metadata['sort'] = 'queryname'
        with open(self.outdir + args.name + '_metadata.json', 'w+') as f:
            json.dump(self.output_metadata, f, indent=4)

def read_pipeline(fasta_file, args):
    mapped_file_pool = persona_ops.m_map_pool(size=0, bound=False, name="mmap_pool")
    #if args.paired:
        #assert(fasta_file.get_shape() == tensor_shape.vector(2))
        #files = tf.unstack(fasta_file) 
        #reader_0 = persona_ops.file_m_map(filename=files[0], pool_handle=mapped_file_pool, 
                                 #synchronous=False, name="file_map_0")
        #reader_1 = persona_ops.file_m_map(filename=files[1], pool_handle=mapped_file_pool,
                                 #synchronous=False, name="file_map_1")
        #queued_results = pipeline.join([reader_0, reader_1], parallel=1, capacity=2, name="read_out")
    #else:
    reader = persona_ops.file_m_map(filename=fasta_file, pool_handle=mapped_file_pool,
                             synchronous=False, name="file_map")
    queued_results = pipeline.join([reader], parallel=1, capacity=2, name="read_out")
    return queued_results[0]

def compress_pipeline(converters, compress_parallelism):
    converted_batch = pipeline.join(converters, parallel=compress_parallelism, capacity=8, multi=True, name="compress_input")

    buf_pool = persona_ops.buffer_pool(size=0, bound=False, name="bufpool")

    for base, meta, first_ord, num_recs in converted_batch:
        base_buf = persona_ops.buffer_pair_compressor(buffer_pool=buf_pool, buffer_pair=base)
        meta_buf = persona_ops.buffer_pair_compressor(buffer_pool=buf_pool, buffer_pair=meta)

        yield base_buf, meta_buf, first_ord, num_recs


def writer_pipeline(compressors, write_parallelism, record_id, output_dir, suffix, args):
    prefix_name = tf.constant("{}_".format(record_id), name="prefix_string")
    compressed_batch = pipeline.join(compressors, parallel=write_parallelism, capacity=8, multi=True, name="write_input")

    for base, meta, first_ordinal, num_recs in compressed_batch:
        first_ord_as_string = string_ops.as_string(first_ordinal, name="first_ord_as_string")
        base_key = string_ops.string_join([output_dir, prefix_name, first_ord_as_string, ".", suffix], name="base_key_string")
        meta_key = string_ops.string_join([output_dir, prefix_name, first_ord_as_string, ".metadata"], name="metadata_key_string")
        base_path = persona_ops.agd_file_system_buffer_writer(record_id=record_id,
                                                     record_type= "text" if args.protein else "base_compact",
                                                     resource_handle=base,
                                                     path=base_key,
                                                     compressed=True,
                                                     first_ordinal=first_ordinal,
                                                     num_records=tf.to_int32(num_recs))
        meta_path = persona_ops.agd_file_system_buffer_writer(record_id=record_id,
                                                     record_type="text",
                                                     resource_handle=meta,
                                                     path=meta_key,
                                                     compressed=True,
                                                     first_ordinal=first_ordinal,
                                                     num_records=tf.to_int32(num_recs))
        yield base_path, meta_path, first_ordinal, num_recs

def conversion_pipeline(queued_fasta, chunk_size, convert_parallelism, args):
    fasta_read_pool = persona_ops.fasta_read_pool(size=0, bound=False, name="fasta_read_pool")

    q = data_flow_ops.FIFOQueue(capacity=32, # big because who cares
                                dtypes=[dtypes.string, dtypes.int64, dtypes.int64],
                                shapes=[tensor_shape.vector(2), tensor_shape.scalar(), tensor_shape.scalar()],
                                name="chunked_output_queue")
    chunker = persona_ops.fasta_chunker(chunk_size=chunk_size, queue_handle=q.queue_ref,
                                        fasta_file=queued_fasta, fasta_pool=fasta_read_pool)

    queue_runner.add_queue_runner(queue_runner.QueueRunner(q, [chunker]))
    bpp = persona_ops.buffer_pair_pool(size=0, bound=False, name="conversion_buffer_pool")
    for _ in range(convert_parallelism):
        fasta_resource, first_ordinal, num_recs = q.dequeue()
        base, meta = persona_ops.agd_fasta_converter(buffer_pair_pool=bpp, input_data=fasta_resource, is_nucleotide=args.dna,
                name="agd_fasta_converter")
        yield base, meta, first_ordinal, num_recs

