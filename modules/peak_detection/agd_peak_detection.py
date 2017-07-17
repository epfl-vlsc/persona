from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.python.ops import data_flow_ops, string_ops

from ..common.service import Service
from common.parse import numeric_min_checker, path_exists_checker
from ..common import parse

import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import queues, pipeline

class CalculateCoverageService(Service):
    """ A class representing a service module in Persona """

    #default inputs
    def get_shortname(self):
        return "coverage"

    def output_dtypes(self):
        return []
    def output_shapes(self):
        return []

    def extract_run_args(self, args):
        dataset = args.dataset
        paths = [ a["path"] for a in dataset["records"] ]
        dataset_dir = args.dataset_dir
        if dataset_dir is None:
            file_path = args.dataset[parse.filepath_key]
            dataset_dir = os.path.dirname(file_path)
        return (os.path.join(dataset_dir, a) for a in paths)

    def add_graph_args(self, parser):
        parser.add_argument("-p", "--parse-parallel", default=1, type=numeric_min_checker(minimum=1, message="read parallelism"),
                            help="total paralellism level for reading data from disk")

        parser.add_argument("-i", "--dataset-dir", type=path_exists_checker(),  help="Directory containing ALL of the chunk files")
        parser.add_argument("-scale", "--scale", default=1,type = int, help="Each coverage value is multiplied by this factor before being reported. Default is 1")
        parser.add_argument("-max", "--max", default=-1, type = int, help="Combine all positions with a depth >= max into a single bin in the histogram")
        parser.add_argument("-bg", "--bg", default=False, action="store_true", help="Report depth in BedGraph format")
        parser.add_argument("-d", "--d", default=False, action="store_true", help="Report the depth at each genome position with 1-based coordinates")
        parser.add_argument("-strand","--strand", default='B', help="Calculate coverage of intervals from a specific strand")
        parser.add_argument("-bga" , "--bga",default=False, action="store_true", help="Report depth in BedGraph format along with zero-entries")
        parser.add_argument("-dz", "--dz" ,default=False, action="store_true", help="Report the depth at each genome position with 0-based coordinates")

    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two
        things: a list of tensors which the runtime will
        evaluate, and a list of run-once ops"""
        # make the graph
        dataset_dir = args.dataset_dir
        if dataset_dir is None:
            file_path = args.dataset[parse.filepath_key]
            dataset_dir = os.path.dirname(file_path)
        if not os.path.exists(dataset_dir) and os.path.isfile(dataset_dir):
            raise EnvironmentError("metadata file '{m}' either doesn't exist or is not a file".format(m=args.dataset))

        if in_queue is None:
            raise EnvironmentError("in queue is none")

        dataset_dir = os.path.dirname(dataset_dir)
        op = agd_peak_detection_local(in_queue=in_queue,
                                       outdir=dataset_dir,
                                       parallel_parse=args.parse_parallel,
                                       scale=args.scale,
                                       argsj = args,
                                       max = args.max,
                                       bg = args.bg,
                                       d = args.d,
                                       strand = args.strand,
                                       bga = args.bga,
                                       dz = args.dz)
        run_once = []
        return [[op]] , run_once



def compress_pipeline(results, compress_parallelism):

    buf_pool = persona_ops.buffer_pool(size=0, bound=False, name="bufpool")

    for buf, num_recs, first_ord, record_id in results:
        compressed_buf = persona_ops.buffer_pair_compressor(buffer_pool=buf_pool, buffer_pair=buf)

        yield compressed_buf, num_recs, first_ord, record_id

def agd_peak_detection_local(in_queue, argsj,outdir=None, parallel_parse=1,scale=1,max=-1,bg=False,d =  False,strand = "B",bga=False,dz= False):
    manifest = argsj.dataset
    if 'reference' not in manifest:
        raise Exception("No reference data in manifest {}. Unaligned BAM not yet supported. Please align dataset first.".format(args.dataset))

    """
    key: tensor with chunk key string
    local_directory: the "base path" from which these should be read
    column_grouping_factor: the number of keys to put together
    parallel_parse: the parallelism for processing records (decomp)
    """

    ref_lens = []
    ref_seqs = []
    for contig in manifest['reference_contigs']:
       ref_lens.append(contig['length'])
       ref_seqs.append(contig['name'])

    parallel_key_dequeue = tuple(in_queue.dequeue() for _ in range(parallel_parse))
    result_chunks = pipeline.local_read_pipeline(upstream_tensors=parallel_key_dequeue, columns=['results'])

    result_chunk_list = [ list(c) for c in result_chunks ]


    parsed_results = pipeline.agd_reader_multi_column_pipeline(upstream_tensorz=result_chunk_list)
    parsed_results_list = list(parsed_results)

    parsed_result = pipeline.join(parsed_results_list, parallel=1, capacity=8, multi=True)[0]



    result_buf, num_results, first_ord, record_id = parsed_result
    result_buf = tf.unstack(result_buf)[0]



    result = persona_ops.agd_peak_detection(results_handle=result_buf, num_records=num_results,ref_sequences=ref_seqs,ref_seq_sizes=ref_lens,scale = scale, max= max,bg = bg,d = d, strand = strand, dz = dz,bga= bga,name="peakdetectionop")

    return result
