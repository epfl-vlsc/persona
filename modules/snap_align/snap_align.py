import os
import json
import multiprocessing
from ..common.service import Service
from common.parse import numeric_min_checker, path_exists_checker, non_empty_string_checker, yes_or_no
from ..common import parse
import glob
import sys

import tensorflow as tf
import itertools

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import queues, pipeline

import logging
logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

class SnapCommonService(Service):
    columns = ["base", "qual"]
    write_columns = []

    def extract_run_args(self, args):
        dataset = args.dataset
        return (a["path"] for a in dataset["records"])

    @staticmethod
    def add_max_secondary(parser):
        parser.add_argument("-s", "--max-secondary", type=numeric_min_checker(0, "must have a non-negative number of secondary results"), default=0, help="Max secondary results to store. >= 0 ")

    def add_run_args(self, parser):
        super().add_run_args(parser=parser)
        self.add_max_secondary(parser=parser)

    def add_graph_args(self, parser):
        # adds the common args to all graphs
        parser.add_argument("-p", "--parallel", type=numeric_min_checker(1, "parallel decompression"), default=2, help="parallel decompression")
        parser.add_argument("-e", "--enqueue", type=numeric_min_checker(1, "parallel enqueuing"), default=1, help="parallel enqueuing / reading from Ceph")
        parser.add_argument("-a", "--aligners", type=numeric_min_checker(1, "number of aligners"), default=1, help="number of aligners")
        parser.add_argument("-t", "--aligner-threads", type=numeric_min_checker(1, "threads per aligner"), default=multiprocessing.cpu_count(), help="the number of threads to use per aligner")
        parser.add_argument("-x", "--subchunking", type=numeric_min_checker(1, "don't go lower than 100 for subchunking size"), default=5000, help="the size of each subchunk (in number of reads)")
        parser.add_argument("-w", "--writers", type=numeric_min_checker(0, "must have a non-negative number of writers"), default=1, help="the number of writer pipelines")
        parser.add_argument("-c", "--compress-parallel", type=int, default=2, help="compress output in parallel. 0 for uncompressed [2]")
        parser.add_argument("--assemblers", default=1, type=numeric_min_checker(1, "must have at least one assembler node"), help="level of parallelism for assembling records")
        parser.add_argument("--deep-verify", default=False, action='store_true', help="verify record integrity")
        parser.add_argument("--paired", default=False, action='store_true', help="interpret dataset as interleaved paired dataset")
        parser.add_argument("-i", "--index-path", type=path_exists_checker(), default="/scratch/stuart/ref_index", help="location of the ref index on all machines. Make sure all machines have this path!")
        self.add_max_secondary(parser=parser)
        parser.add_argument("--snap-args", type=str, default="", help="SNAP algorithm specific args. Pass with enclosing \" \". E.g. \"-om 5 -omax 1\" . See SNAP documentation for all options.")

    def make_central_pipeline(self, args, input_gen, pass_around_gen):
        
        self.write_columns.append('results')
        for i in range(args.max_secondary):
            self.write_columns.append('secondary{}'.format(i))

        self.write_columns = [ {"type": "structured", "extension": a} for a in self.write_columns]

        joiner = tuple(tuple(a) + tuple(b) for a,b in zip(input_gen, pass_around_gen))
        ready_to_process = pipeline.join(upstream_tensors=joiner,
                                         parallel=args.parallel,
                                         capacity=args.parallel, # multiplied by some factor?
                                         multi=True, name="ready_to_process")
        # need to unpack better here
        to_agd_reader, pass_around_agd_reader = zip(*((a[:2], a[2:]) for a in ready_to_process))
        multi_column_gen = pipeline.agd_reader_multi_column_pipeline(upstream_tensorz=to_agd_reader)

        def process_processed_bufs():
            for processed_column, pass_around in zip(multi_column_gen, pass_around_agd_reader):
                if isinstance(pass_around, tf.Tensor):
                    pass_around = (pass_around,)
                yield tuple(a for a in itertools.chain(processed_column, pass_around))

        processed_bufs = tuple(a for a in process_processed_bufs())
        ready_to_assemble = pipeline.join(upstream_tensors=processed_bufs,
                                          parallel=args.assemblers,
                                          capacity=args.assemblers*2, multi=True, name="ready_to_assemble") # TODO these params are kinda arbitrary :/
        # ready_to_assemble: [output_buffers, num_records, first_ordinal, record_id, pass_around {flattened}) x N]
        to_assembler, pass_around_assembler = zip(*((a[:2], a[1:]) for a in ready_to_assemble))
        # each item out of this is a handle to AGDReads
        agd_read_assembler_gen = tuple(pipeline.agd_read_assembler(upstream_tensors=to_assembler, include_meta=False))
        # assembled_records, ready_to_align: [(agd_reads_handle, (num_records, first_ordinal, record_id), (pass_around)) x N]
        assembled_records_gen = tuple(zip(agd_read_assembler_gen, pass_around_assembler))
        assembled_records = tuple((a,) + tuple(b) for a,b in assembled_records_gen)
        ready_to_align = pipeline.join(upstream_tensors=assembled_records,
                                       parallel=args.aligners, capacity=int(args.aligners*1.5), multi=True, name="ready_to_align") # TODO still have default capacity here :/

        if args.paired:
            aligner_type = persona_ops.snap_align_paired
            aligner_options = persona_ops.paired_aligner_options(cmd_line=args.snap_args.split(), name="paired_aligner_options")
            executor_type = persona_ops.snap_paired_executor
        else:
            aligner_type = persona_ops.snap_align_single
            aligner_options = persona_ops.aligner_options(cmd_line=args.snap_args.split(), name="aligner_options") # -o output.sam will not actually do anything
            executor_type = persona_ops.snap_single_executor

        first_assembled_result = ready_to_align[0][1:]
        sink_queue_shapes = [a.get_shape() for a in first_assembled_result]
        sink_queue_dtypes = [a.dtype for a in first_assembled_result]

        aligner_dtype = tf.string
        aligner_shape = (args.max_secondary+1, 2)
        sink_queue_shapes.append(aligner_shape)
        sink_queue_dtypes.append(aligner_dtype)

        pass_around_aligners = tuple(a[1:] for a in ready_to_align) # type: [(num_records, first_ordinal, record_id, pass_around x N) x N]
        pass_to_aligners = tuple(a[0] for a in ready_to_align)

        buffer_list_pool = persona_ops.buffer_list_pool(**pipeline.pool_default_args)
        genome = persona_ops.genome_index(genome_location=args.index_path, name="genome_loader")

        def make_aligners():
            single_executor = executor_type(num_threads=args.aligner_threads,
                                            work_queue_size=args.aligners+1,
                                            options_handle=aligner_options,
                                            genome_handle=genome)
            for read_handle, pass_around in zip(pass_to_aligners, pass_around_aligners):
                aligner_results = aligner_type(read=read_handle,
                                               buffer_list_pool=buffer_list_pool,
                                               subchunk_size=args.subchunking,
                                               executor_handle=single_executor,
                                               max_secondary=args.max_secondary)
                yield (aligner_results,) + tuple(pass_around)

        aligners = tuple(make_aligners())
        # aligners: [(buffer_list_handle, num_records, first_ordinal, record_id, pass_around X N) x N], that is COMPLETELY FLAT
        if args.compress_parallel > 0:
            aligner_results_to_compress = pipeline.join(upstream_tensors=aligners, parallel=args.compress_parallel, multi=True, capacity=4, name="ready_to_compress")
            to_compressors = (a[0] for a in aligner_results_to_compress)
            around_compressors = (a[1:] for a in aligner_results_to_compress)
            compressed_buffers = pipeline.aligner_compress_pipeline(upstream_tensors=to_compressors)
            after_compression = ((a,)+tuple(b) for a,b in zip(compressed_buffers, around_compressors))
            aligners = tuple(after_compression)

        aligned_results = pipeline.join(upstream_tensors=aligners, parallel=args.writers,
                                        multi=True, capacity=4, name="aligned_results")

        ref_seqs, lens = persona_ops.snap_index_reference_sequences(genome_handle=genome)
        # Taking this out because it currently breaks distributed runtime
        return aligned_results, (genome, ref_seqs, lens) # returns [(buffer_list_handle, num_records, first_ordinal, record_id, pass_around X N) x N], that is COMPLETELY FLAT

    def on_finish(self, args, results):
        columns = args.dataset['columns']
        if "results" not in columns:
            columns.append('results')
        for i in range(args.max_secondary):
            to_add = "secondary{}".format(i)
            if to_add not in columns:
                columns.append(to_add)
        with open(args.dataset[parse.filepath_key], 'w+') as f:
            args.dataset.pop(parse.filepath_key, None)  # we dont need to write the actual file path out
            json.dump(args.dataset, f, indent=4)

class CephCommonService(SnapCommonService):

    def input_dtypes(self, args):
        """ Ceph services require the key and the namespace name """
        return (tf.string,) * 2

    def input_shapes(self, args):
        """ Ceph services require the key and the namespace name """
        return (tf.TensorShape([]),) * 2

    def extract_run_args(self, args):
        dataset = args.dataset
        namespace_key = "namespace"
        if namespace_key in dataset:
            namespace = dataset[namespace_key]
        else:
            namespace = dataset['name']
            log.warning("Namespace key '{n}' not in dataset file. defaulting to dataset name '{dn}'".format(n=namespace_key, dn=namespace))
        return ((a, namespace) for a in super().extract_run_args(args=args))

    def add_graph_args(self, parser):
        super().add_graph_args(parser=parser)
        parser.add_argument("--ceph-cluster-name", type=non_empty_string_checker, default="ceph", help="name for the ceph cluster")
        parser.add_argument("--ceph-user-name", type=non_empty_string_checker, default="client.dcsl1024", help="ceph username")
        parser.add_argument("--ceph-conf-path", type=path_exists_checker(check_dir=False), default="/etc/ceph/ceph.conf", help="path for the ceph configuration")
        parser.add_argument("--ceph-read-chunk-size", default=(2**26), type=numeric_min_checker(128, "must have a reasonably large minimum read size from Ceph"), help="minimum size to read from ceph storage, in bytes")
        parser.add_argument("--ceph-pool-name", help="override the pool name to use (if specified or not in the json file")

class CephSnapService(CephCommonService):
    """ A service to use the snap aligner with a ceph dataset """

    def get_shortname(self):
        return "ceph"

    def output_dtypes(self, args):
        return ((tf.string,) * 2) + (tf.int32, tf.int64) + ((tf.string,) * 2)

    def output_shapes(self, args):
        return (tf.TensorShape([]),) * 5 + (tf.TensorShape((args.max_secondary+1,)),)

    def make_graph(self, in_queue, args):
        """
        :param in_queue: 
        :param args: 
        :return: [ key, namespace, num_records, first_ordinal, record_id, full_path ]
        """
        if args.ceph_pool_name is None:
            if not hasattr(args, "dataset"):
                raise Exception("Must specify pool name manually if dataset isn't specified")
            dataset = args.dataset
            pool_key = "ceph_pool"
            if pool_key not in dataset:
                raise Exception("Please provide a dataset that has the pool specified with '{pk}', or specify with --ceph-pool-name when launching")
            args.ceph_pool_name = dataset[pool_key]

        parallel_key_dequeue = (tf.unstack(in_queue.dequeue()) for _ in range(args.enqueue))
        # parallel_key_dequeue = [(key, namespace) x N]
        ceph_read_buffers = tuple(pipeline.ceph_read_pipeline(upstream_tensors=parallel_key_dequeue,
                                                              user_name=args.ceph_user_name,
                                                              cluster_name=args.ceph_cluster_name,
                                                              ceph_conf_path=args.ceph_conf_path,
                                                              pool_name=args.ceph_pool_name,
                                                              columns=self.columns))
        pass_around_central_gen = (a[:2] for a in ceph_read_buffers) # key, namespace
        to_central_gen = (a[2] for a in ceph_read_buffers) # [ chunk_buffer_handles x N ]

        # aligner_results: [(buffer_list_result, (num_records, first_ordinal, record_id), (key, namespace)) x N]
        aligner_results, run_first = self.make_central_pipeline(args=args,
                                                                input_gen=to_central_gen,
                                                                pass_around_gen=pass_around_central_gen)

        to_writer_gen = ((key, namespace, num_records, first_ordinal, record_id, buffer_list_ref) for buffer_list_ref, num_records, first_ordinal, record_id, key, namespace in aligner_results)
        # ceph writer pipeline wants (key, first_ord, num_recs, namespace, record_id, column_handle)
        # type of aligned_results_queue: [(key, namespace, num_records, first_ordinal, record_id, buffer_list_ref) x N]
        # this happens to match the iterator in ceph_aligner_write_pipeline, but otherwise you can mix like above
        writer_outputs = (tuple(a) for a in pipeline.ceph_aligner_write_pipeline(
            upstream_tensors=to_writer_gen,
            user_name=args.ceph_user_name,
            cluster_name=args.ceph_cluster_name,
            ceph_conf_path=args.ceph_conf_path,
            compressed=(args.compress_parallel > 0),
            pool_name=args.ceph_pool_name
        ))
        output_tensors = (b+(a,) for a,b in zip(writer_outputs, ((key, namespace, num_records, first_ordinal, record_id)
                                                       for _, num_records, first_ordinal, record_id, key, namespace in aligner_results)))
        output_tensors = tuple(output_tensors)
        return output_tensors, run_first

class LocalCommonService(SnapCommonService):
    def extract_run_args(self, args):
        dataset_dir = args.dataset_dir
        if dataset_dir is None:
            file_path = args.dataset[parse.filepath_key]
            dataset_dir = os.path.dirname(file_path)
       
        if 'results' in args.dataset['columns']:
            if not yes_or_no("This dataset appears to be aligned. Do you want to realign it (deletes existing results) "):
                sys.exit(0)
            else:
                for f in glob.glob(dataset_dir + "/*.results"):
                    os.remove(f)
                for f in glob.glob(dataset_dir + "/*.secondary*"):
                    os.remove(f)

        return (os.path.join(dataset_dir, a) for a in super().extract_run_args(args=args))

    def add_run_args(self, parser):
        super().add_run_args(parser=parser)
        parser.add_argument("-d", "--dataset-dir", type=path_exists_checker(), help="Directory containing ALL of the chunk files")
    
    def on_finish(self, args, results):
        # add results column to metadata
        # add reference data to metadata
        # TODO do the same thing for the ceph version

        columns = args.dataset['columns']
        _, ref_seqs, lens = results[0]
        #print(ref_seqs.decode("utf-8"))
        lens = lens.decode("utf-8").split('|')
        ref_list = []
        for i, ref in enumerate(ref_seqs.decode("utf-8").split('|')):
            ref_list.append({'name':ref, 'length':lens[i], 'index':i})
        args.dataset['reference_contigs'] = ref_list
        args.dataset['reference'] = args.index_path

        if "results" not in columns:
            columns.append('results')
        for i in range(args.max_secondary):
            to_add = "secondary{}".format(i)
            if to_add not in columns:
                columns.append(to_add)
        args.dataset['columns'] = columns

        with open(args.dataset[parse.filepath_key], 'w+') as f:
            args.dataset.pop(parse.filepath_key, None)  # we dont need to write the actual file path out
            json.dump(args.dataset, f, indent=4)

class LocalSnapService(LocalCommonService):
    """ A service to use the SNAP aligner with a local dataset """

    def get_shortname(self):
        return "local"

    def output_dtypes(self, args):
        return (tf.string, tf.int32, tf.int64, tf.string, tf.string)

    def output_shapes(self, args):
        return (tf.TensorShape([]),) * 4 + (tf.TensorShape([args.max_secondary+1]),)

    def make_graph(self, in_queue, args):
        parallel_key_dequeue = tuple(in_queue.dequeue() for _ in range(args.enqueue))
        # read_files: [(file_path, (mmaped_file_handles, a gen)) x N]
        read_files = tuple((path_base,) + tuple(read_gen) for path_base, read_gen in zip(parallel_key_dequeue, pipeline.local_read_pipeline(upstream_tensors=parallel_key_dequeue, columns=self.columns)))
        # need to use tf.tuple to make sure that these are both made ready at the same time
        to_central_gen = (a[1:] for a in read_files)
        pass_around_gen = ((a[0],) for a in read_files)

        aligner_results, run_first = tuple(self.make_central_pipeline(args=args,
                                                                      input_gen=to_central_gen,
                                                                      pass_around_gen=pass_around_gen))

        to_writer_gen = tuple((buffer_list_handle, record_id, first_ordinal, num_records, file_basename) for buffer_list_handle, num_records, first_ordinal, record_id, file_basename in aligner_results)
        written_records = (tuple(a) for a in pipeline.local_write_pipeline(upstream_tensors=to_writer_gen, compressed=(args.compress_parallel > 0), record_types=self.write_columns))
        final_output_gen = zip(written_records, ((record_id, num_records, first_ordinal, file_basename) for _, num_records, first_ordinal, record_id, file_basename in aligner_results))
        output = (b+(a,) for a,b in final_output_gen)
        return output, run_first


