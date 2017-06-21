
import tensorflow as tf
import shutil
import sys
import os
import errno
import json
import pysam
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

class ImportBamService(Service):
   
    #default inputs
    def get_shortname(self):
        return "import_fastq"

    def add_graph_args(self, parser):

        # TODO sane defaults depending on num schedulable cores
        parser.add_argument("-c", "--chunk", type=numeric_min_checker(1, "chunk size"), default=100000, help="chunk size to create records")
        parser.add_argument("-p", "--parallel-conversion", type=numeric_min_checker(1, "parallel conversion"), default=1, help="number of parallel converters")
        parser.add_argument("-n", "--name", required=True, help="name for the record")
        parser.add_argument("-o", "--out", default=".", help="directory to write the final record to")
        parser.add_argument("-w", "--write", default=1, type=numeric_min_checker(1, "write parallelism"), help="number of parallel writers")
        parser.add_argument("--unaligned", default=False, action='store_true', help="Set true if BAM file is unaligned")
        parser.add_argument("--compress-parallel", default=1, type=numeric_min_checker(1, "compress parallelism"), help="number of parallel compression pipelines")
        parser.add_argument("bam_file", help="the fastq file to convert")

    def add_run_args(self, parser):
        pass

    def distributed_capability(self):
        return False

    def output_dtypes(self, args):
        return []

    def output_shapes(self, args):
        return []

    def extract_run_args(self, args):
        # nothing needed here
        return []

    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two 
        things: a list of tensors which the runtime will 
        evaluate, and a list of run-once ops"""
        # make the graph

        self.outdir = os.path.abspath(args.out) + '/'
        mkdir_p(self.outdir)
        print("BAM import creating new dataset in {}".format(self.outdir))

        self.output_records = []
        columns = ['base', 'qual', 'metadata'] if args.unaligned else ['base', 'qual', 'metadata', 'results']
        self.output_metadata = {
            "name": args.name, "version": 1, "records": self.output_records, "columns": columns
        }

        #import ipdb; ipdb.set_trace()
        bamfile = pysam.AlignmentFile(args.bam_file, "rb")
        ref_lens = bamfile.lengths
        ref_names = bamfile.references
        refs = []
        for i, item in enumerate(list(zip(ref_names, ref_lens))):
            name, length = item
            refs.append({'index':i, 'length':length, 'name':name})
        self.output_metadata['reference_contigs'] = refs
        self.output_metadata['sort'] = bamfile.header['HD']['SO']
        print("bamfile sort order is {}".format(self.output_metadata['sort']))

        bpp = persona_ops.buffer_pair_pool(size=0, bound=False, name="bufpool")
        chunk, num_recs, first_ord = persona_ops.agd_import_bam(path=args.bam_file, num_threads=10, 
            ref_seq_lens=ref_lens, chunk_size=args.chunk, unaligned=args.unaligned, bufpair_pool=bpp)

        compressors = tuple(compress_pipeline(converters=[[chunk, first_ord, num_recs]], compress_parallelism=args.compress_parallel))

        writers = tuple(writer_pipeline(compressors, args.write, args.name, self.outdir))
        #final = pipeline.join(upstream_tensors=writers, capacity=8, parallel=1, multi=True)[0]

        return writers, []
    
    def on_finish(self, args, results):
        for res in results:
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

