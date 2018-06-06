from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.persona import pipeline

import os
import json
import resource
import multiprocessing
import sys
from tensorflow.python.ops import data_flow_ops, string_ops
from argparse import ArgumentTypeError
from . import environments
import shutil

from ..common.service import Service
from common.parse import numeric_min_checker, yes_or_no
import glob

import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()


class ProteinClusterService(Service):
    """ A class representing a service module in Persona """

    # default inputs
    def get_shortname(self):
        return "protcluster"

    def output_dtypes(self, args):
        return []

    def output_shapes(self, args):
        return []

    def input_dtypes(self, args):
        # the path to the chunk, and how many prots are in that genome
        # and abs seq of each chunk
        return [tf.string, tf.int32, tf.int32]

    def input_shapes(self, args):
        return [tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])]

    def extract_run_args(self, args):
        if len(args.datasets) != len(set(args.datasets)):
            raise ArgumentTypeError("Duplicate datasets on command line!")

        datasets = [(os.path.dirname(x), json.load(open(x)))
                    for x in args.datasets]
        self.chunk_size = datasets[0][1]["records"][0]["last"] - datasets[0][1]["records"][0]["first"]
        self.total_chunks = 0
        print("chunk size is {}".format(self.chunk_size))
        paths = []
        num_seqs = []
        for t in datasets:
            if t[1]["records"][0]["last"] - t[1]["records"][0]["first"] != self.chunk_size:
                raise ArgumentTypeError(
                    "Datasets do not have equal chunk sizes")
            total_seqs = 0
            for p in t[1]["records"]:
                paths.append(os.path.join(t[0], p["path"]))
                self.total_chunks += 1
                total_seqs += p["last"] - p["first"]

            for x in t[1]["records"]:
                num_seqs.append(total_seqs)

        abs_seq = [x for x in range(len(paths))]
        assert (len(abs_seq) == len(paths))
        print(abs_seq)
        return [paths, num_seqs, abs_seq]

    def add_run_args(self, parser):
        def dataset_parser(filename):
            if not os.path.isfile(filename):
                raise ArgumentTypeError(
                    "AGD metadata file not present at {}".format(filename))
            return filename

        parser.add_argument(
            "datasets",
            type=dataset_parser,
            nargs='+',
            help="Directories containing ALL of the chunk files")

    def add_graph_args(self, parser):
        parser.add_argument(
            "-p",
            "--parse-parallel",
            default=1,
            type=numeric_min_checker(minimum=1, message="read parallelism"),
            help="total paralellism level for reading data from disk")
        parser.add_argument(
            "-w",
            "--write-parallel",
            default=1,
            help="number of writers to use",
            type=numeric_min_checker(
                minimum=1, message="number of writers min"))
        parser.add_argument(
            "-n",
            "--nodes",
            default=1,
            help="number of ring nodes",
            type=numeric_min_checker(minimum=1, message="number of nodes min"))
        parser.add_argument(
            "-l",
            "--cluster-length",
            default=100,
            help="Dim 0 size of cluster tensors",
            type=numeric_min_checker(
                minimum=1,
                message="Dim 0 size of cluster tensors must be > 0"))
        parser.add_argument(
            "-t",
            "--num-threads",
            default=multiprocessing.cpu_count(),
            help="Number of threads in alignment executor.",
            type=numeric_min_checker(
                minimum=1,
                message="Num threads must be > 0"))

        parser.add_argument(
            "-c", "--config", default="./params.json", help="JSON config file")
        parser.add_argument(
            "-o",
            "--output-dir",
            default="matches_out",
            help="output file dir")
        parser.add_argument(
            "--do-allall",
            action='store_true',
            help=
            "Do the intra cluster all all. Leave out for testing clustering timing."
        )

    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two
        things: a list of tensors which the runtime will
        evaluate, and a list of run-once ops"""

        print("setting rlimit")
        # required to avoid stack overflows caused by alignment functions
        # that stack allocate big arrays
        resource.setrlimit(resource.RLIMIT_STACK, (65532000, 65532000))
        # make the graph
        args.config = json.load(open(args.config))

        if in_queue is None:
            raise EnvironmentError("in queue is none")

        ops = self.agd_protein_cluster_local(in_queue, args)
        run_once = []
        print(ops)
        print(os.getpid())
        #import ipdb; ipdb.set_trace()
        return ops, run_once

    def make_ring(self, args, num_nodes, input_queue, envs, shapes, types,
                  candidate_map, executor):

        left_queue = None
        ops = []
        cluster_tensors_out = []

        prev_nb_queue = None

        for i in range(num_nodes):
            op, nb, nbo, cto = self.make_ring_node(args, input_queue, envs, i,
                                                   shapes, types,
                                                   candidate_map, executor)
            ops.append([op])
            cluster_tensors_out.append(cto)
            if i == 0:
                left_queue = nb
            if prev_nb_queue:
                enq = nb.enqueue(prev_nb_queue.dequeue())
                tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(nb, [enq]))

            prev_nb_queue = nbo

            if i == num_nodes - 1:
                print("adding last neighbor queue runner")
                enq = left_queue.enqueue(nbo.dequeue())
                tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(left_queue, [enq]))

        print(cluster_tensors_out)
        match_ints, match_doubles, genomes = pipeline.join(
            [x for x in cluster_tensors_out],
            parallel=1,
            capacity=32,
            multi=True)[0]

        if os.path.exists(args.output_dir):
            if not yes_or_no("This output dir exists. Overwrite?"):
                sys.exit(0)
            else:
                for f in glob.glob(args.output_dir + "/*"):
                    shutil.rmtree(f)
        else:
            os.mkdir(args.output_dir)

        agg = persona_ops.agd_cluster_aggregate(
            match_ints=match_ints,
            match_doubles=match_doubles,
            genomes=genomes,
            output_dir=args.output_dir)

        #ops.append(cluster_tensor_out)
        print(ops)
        print(agg)
        return ops + [[agg]]

    def make_ring_node(self, args, input_queue, envs, node_id, shapes, types,
                       candidate_map, executor):
        # output (op, neighbor_queue, neighbor_queue_out, cluster_tensor_out)
        nb_q = data_flow_ops.FIFOQueue(
            capacity=20,  # TODO capacity
            dtypes=types,
            shapes=shapes,
            name="nb_" + str(node_id))

        nb_q_o = data_flow_ops.FIFOQueue(
            capacity=20,  # TODO capacity
            dtypes=types,
            shapes=shapes,
            name="nbo_" + str(node_id))

        s = args.cluster_length
        c_q = data_flow_ops.FIFOQueue(
            capacity=30,  # TODO capacity
            dtypes=[tf.int32, tf.float64, tf.string],
            shapes=[
                tf.TensorShape([s, 6]),
                tf.TensorShape([s, 3]),
                tf.TensorShape([s, 2])
            ])

        cluster_tensor_out = c_q.dequeue()

        should_seed = True  # if node_id == 0 else False

        op = persona_ops.agd_protein_cluster(
            input_queue=input_queue.queue_ref,
            neighbor_queue=nb_q.queue_ref,
            neighbor_queue_out=nb_q_o.queue_ref,
            cluster_queue=c_q.queue_ref,
            alignment_envs=envs,
            node_id=node_id,
            ring_size=args.nodes,
            min_score=args.config["min_score"],
            max_reps=args.config["max_reps"],
            max_n_aa_not_covered=args.config["max_n_aa_not_covered"],
            subsequence_homology=args.config["subsequence_homology"],
            total_chunks=self.total_chunks,
            chunk_size=self.chunk_size,
            should_seed=should_seed,
            cluster_length=args.cluster_length,
            candidate_map=candidate_map,
            executor=executor,
            do_allall=args.do_allall,
            name="protclustop")

        return (op, nb_q, nb_q_o, cluster_tensor_out)

    def agd_protein_cluster_local(self,
                                  in_queue,
                                  args,
                                  parallel_parse=1,
                                  parallel_write=1,
                                  parallel_compress=1):
        """
        key: tensor with chunk key string
        local_directory: the "base path" from which these should be read
        column_grouping_factor: the number of keys to put together
        parallel_parse: the parallelism for processing records (decomp)
        """

        parallel_key_dequeue = tuple(
            in_queue.dequeue() for _ in range(parallel_parse))
        keys = [x[0] for x in parallel_key_dequeue]
        print("key is {}".format(keys))
        total_recs = [x[1] for x in parallel_key_dequeue]
        abs_seq = [x[2] for x in parallel_key_dequeue]
        result_chunks = pipeline.local_read_pipeline(
            upstream_tensors=keys, columns=['prot'])

        result_chunk_list = [list(c) for c in result_chunks]

        parsed_results = pipeline.agd_reader_multi_column_pipeline(
            upstream_tensorz=result_chunk_list)
        #print("parsed result is {}".format(parsed_results))
        parsed_results_list = [list(x) for x in parsed_results]
        for i, x in enumerate(parsed_results_list):
            x.append(total_recs[i])
            x.append(abs_seq[i])
        #print("parsed result list is {}".format(parsed_results_list))

        parsed_result = pipeline.join(
            parsed_results_list, parallel=1, capacity=8, multi=True)[0]

        # result_buf, num_recs, first_ord, record_id
        #parsed_results = tf.contrib.persona.persona_in_pipe(key=key, dataset_dir=local_directory, columns=["results"], parse_parallel=parallel_parse,
        #process_parallel=1)

        print(parsed_result)
        result_buf, num_recs, first_ord, record_id, total_recs, abs_seq = parsed_result
        result_buf = tf.unstack(result_buf)[0]
        print(result_buf)

        protein_tensor = persona_ops.agd_chunk_to_tensor(result_buf)
        # form main input queue with protein_tensor, num_recs, sequence, was_added, coverages, record_id (genome_name), first_ord
        shapes = [
            protein_tensor.shape, num_recs.shape,
            tf.TensorShape([]),
            tf.TensorShape([self.chunk_size]),
            tf.TensorShape([self.chunk_size]),
            tf.TensorShape([]), first_ord.shape, total_recs.shape,
            tf.TensorShape([])
        ]
        types = [
            protein_tensor.dtype.base_dtype, num_recs.dtype, tf.int32, tf.bool,
            tf.string, tf.string, first_ord.dtype, total_recs.dtype, tf.int32
        ]
        q = data_flow_ops.FIFOQueue(
            capacity=2,  # TODO capacity
            dtypes=types,
            shapes=shapes)

        print(shapes)
        seq = tf.constant(0)
        was_added = tf.constant([False for x in range(self.chunk_size)])
        coverages = tf.constant(["" for x in range(self.chunk_size)])
        enq = q.enqueue([
            protein_tensor, num_recs, seq, was_added, coverages, record_id,
            first_ord, total_recs, abs_seq
        ])

        tf.train.queue_runner.add_queue_runner(
            tf.train.queue_runner.QueueRunner(q, [enq]))

        envs = environments.make_envs_op()

        candidate_map = persona_ops.candidate_map()
        print("Threads is {}".format(args.num_threads))
        executor = persona_ops.alignment_executor(num_threads=args.num_threads, capacity=100)
        all_ops = self.make_ring(args, args.nodes, q, envs, shapes, types,
                                 candidate_map, executor)

        #result_to_write = pipeline.join([result_out, num_results, first_ord, record_id], parallel=parallel_write,
        #capacity=8, multi=False)

        #compressed = compress_pipeline(result_to_write, parallel_compress)

        #written = _make_writers(compressed_batch=list(compressed), output_dir=outdir, write_parallelism=parallel_write)

        #recs = list(written)
        #all_written_keys = pipeline.join(recs, parallel=1, capacity=8, multi=False)

        return all_ops


def _make_writers(compressed_batch, output_dir, write_parallelism):

    compressed_single = pipeline.join(
        compressed_batch, parallel=write_parallelism, capacity=8, multi=True)

    for buf, num_recs, first_ordinal, record_id in compressed_single:

        first_ord_as_string = string_ops.as_string(
            first_ordinal, name="first_ord_as_string")
        result_key = string_ops.string_join(
            [output_dir, "/", record_id, "_", first_ord_as_string, ".results"],
            name="base_key_string")

        result = persona_ops.agd_file_system_buffer_writer(
            record_id=record_id,
            record_type="structured",
            resource_handle=buf,
            path=result_key,
            compressed=True,
            first_ordinal=first_ordinal,
            num_records=tf.to_int32(num_recs))

        yield result  # writes out the file path key (full path)


def compress_pipeline(results, compress_parallelism):

    buf_pool = persona_ops.buffer_pool(size=0, bound=False, name="bufpool")

    for buf, num_recs, first_ord, record_id in results:
        compressed_buf = persona_ops.buffer_pair_compressor(
            buffer_pool=buf_pool, buffer_pair=buf)

        yield compressed_buf, num_recs, first_ord, record_id
