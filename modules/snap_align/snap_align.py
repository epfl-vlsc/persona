import os
import json
import multiprocessing
from ..common.service import Service
from common.parse import numeric_min_checker, path_exists_checker, non_empty_string_checker
from ..common import parse

import tensorflow as tf
import itertools

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import queues, pipeline

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
        parser.add_argument("-c", "--compress", default=False, action='store_true', help="compress the output")
        parser.add_argument("--assemblers", default=1, type=numeric_min_checker(1, "must have at least one assembler node"), help="level of parallelism for assembling records")
        parser.add_argument("--compress-parallel", default=1, type=numeric_min_checker(1, "must have at least one parallel compressor"), help="the parallelism for the output compression pipeline, if set")
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
        if args.compress:
            aligner_results_to_compress = pipeline.join(upstream_tensors=aligners, parallel=args.compress_parallel, multi=True, capacity=32, name="ready_to_compress")
            to_compressors = (a[0] for a in aligner_results_to_compress)
            around_compressors = (a[1:] for a in aligner_results_to_compress)
            compressed_buffers = pipeline.aligner_compress_pipeline(upstream_tensors=to_compressors)
            after_compression = ((a,)+tuple(b) for a,b in zip(compressed_buffers, around_compressors))
            aligners = tuple(after_compression)

        aligned_results = pipeline.join(upstream_tensors=aligners, parallel=args.writers,
                                        multi=True, capacity=32, name="aligned_results")

        ref_seqs, lens = persona_ops.snap_index_reference_sequences(genome_handle=genome)
        return aligned_results, (genome, ref_seqs, lens) # returns [(buffer_list_handle, num_records, first_ordinal, record_id, pass_around X N) x N], that is COMPLETELY FLAT

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
        namespace = dataset.get(namespace_key, "")
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
            compressed=args.compress,
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
        ref_list = []
        for i, ref in enumerate(ref_seqs):
            ref_list.append({'name':ref.decode("utf-8"), 'length':lens[i].item(), 'index':i})
        args.dataset['reference_contigs'] = ref_list
        args.dataset['reference'] = args.index_path

        if "results" not in columns:
            columns.append('results')
        for i in range(args.max_secondary):
            to_add = "secondary{}".format(i)
            if to_add not in columns:
                columns.append(to_add)
        args.dataset['columns'] = columns

        for metafile in os.listdir(args.dataset_dir):
            if metafile.endswith(".json"):
                with open(os.path.join(args.dataset_dir, metafile), 'w+') as f:
                    json.dump(args.dataset, f, indent=4)
                break

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
        written_records = (tuple(a) for a in pipeline.local_write_pipeline(upstream_tensors=to_writer_gen, compressed=args.compress, record_types=self.write_columns))
        final_output_gen = zip(written_records, ((record_id, num_records, first_ordinal, file_basename) for _, num_records, first_ordinal, record_id, file_basename in aligner_results))
        output = (b+(a,) for a,b in final_output_gen)
        return output, run_first

class OldCephSnapService(Service):

    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two 
        things: a list of tensors which the runtime will 
        evaluate, and a list of run-once ops"""
        # make the graph

        key = in_queue.dequeue()
        ops, run_once = run(key, args)

        return ops, run_once

def build_queues(key, parallel_enqueue):
    record_queue = tf.contrib.persona.batch_pdq(tensor_list=[key],
                                      batch_size=1, capacity=3, enqueue_many=False,
                                      num_threads=1, num_dq_ops=parallel_enqueue,
                                      name="chunk_keys")
    return record_queue


def create_ops(processed_batch, deep_verify, num_aligners, aligner_threads, subchunk_size, num_writers, index_path, null_align, paired, max_secondary):

    genome = persona_ops.genome_index(genome_location=index_path, name="genome_loader")
    if paired:
        options = persona_ops.paired_aligner_options(cmd_line="-o output.sam".split(), name="paired_aligner_options")
    else:
        options = persona_ops.aligner_options(cmd_line="-o output.sam -om 20 -omax 1".split(), name="aligner_options") # -o output.sam will not actually do anything

    pp = persona_ops.buffer_pool(size=1, bound=False)
    drp = persona_ops.agd_read_pool(size=0, bound=False)

    aggregate_enqueue = []
    for chunk in processed_batch:
        key = chunk[0]
        num_reads = chunk[1]
        first_ord = chunk[2]
        base_reads = chunk[3]
        qual_reads = chunk[4]

        agd_read = persona_ops.no_meta_agd_assembler(agd_read_pool=drp,
                                                  base_handle=base_reads,
                                                  qual_handle=qual_reads,
                                                  num_records=num_reads, name="agd_assembler")

        aggregate_enqueue.append((agd_read, key, num_reads, first_ord))

    agd_records = tf.contrib.persona.batch_join_pdq(tensor_list_list=[e for e in aggregate_enqueue],
                                          batch_size=1, capacity=num_aligners + 2,
                                          enqueue_many=False,
                                          num_dq_ops=num_aligners,
                                          name="agd_reads_to_aligners")
    results = []
    blp = persona_ops.buffer_list_pool(size=1, bound=False)

    extra_aligners = aligner_threads % num_aligners
    threads_per_aligner = aligner_threads // num_aligners

    for i, (record, key, num_records, first_ordinal) in enumerate(agd_records):
        thread_count = threads_per_aligner + 1 if i < extra_aligners else threads_per_aligner
        if null_align is not None:
            aligned_result = persona_ops.null_aligner(buffer_list_pool=blp, read=record,
                                                     subchunk_size=subchunk_size, extra_wait=null_align, name="null_aligner")
        else:
            if paired:
                aligner_type = persona_ops.snap_align_paired
                name="snap_paired_aligner"
            else:
                aligner_type = persona_ops.snap_align_single
                name="snap_single_aligner"
            aligned_result = aligner_type(genome_handle=genome, options_handle=options, num_threads=thread_count,
                                          buffer_list_pool=blp, read=record, subchunk_size=subchunk_size, 
                                          max_secondary=max_secondary, name=name)

        result = (aligned_result, key, num_records, first_ordinal)
        results.append(result)

    aligned_results = tf.contrib.persona.batch_join_pdq(tensor_list_list=[r for r in results],
                                              batch_size=1, capacity=2*num_aligners + 10, #this queue should always be empty!
                                              num_dq_ops=max(1, num_writers), name="results_to_sink")
    return aligned_results, genome, options

def local_write_results(aligned_results, output_path, record_name, compress_output):
    ops = []
    for result_out, key_out, num_records, first_ordinal in aligned_results:
        writer_op = persona_ops.parallel_column_writer(
            column_handle=result_out,
            record_type="results",
            record_id=record_name,
            num_records=num_records,
            first_ordinal=first_ordinal,
            file_path=key_out, name="results_file_writer",
            compress=compress_output, output_dir=output_path
        )
        ops.append(writer_op)
    return ops

def ceph_write_results(aligned_results, record_name, compress_output, cluster_name, user_name, pool_name, ceph_conf_path):
    ops = []
    for result_out, key_out, num_records, first_ordinal in aligned_results:
        key_passthrough = persona_ops.ceph_writer(
            cluster_name=cluster_name,
            user_name=user_name,
            pool_name=pool_name,
            ceph_conf_path=ceph_conf_path,
            compress=compress_output,
            record_id=record_name,
            record_type="results",
            column_handle=result_out,
            file_name=key_out,
            first_ordinal=first_ordinal,
            num_records=num_records,
            name="Ceph_Writer")
        ops.append(key_passthrough)
    return ops


def run(key, args):
    genomes = []

    parallel_enqueue = args.enqueue; parallel_dequeue = args.parallel
    mmap_queue_length = args.mmap_queue; min_ceph_read_size = args.ceph_read_chunk_size
    summary = args.summary
    ceph_pool_name = args.ceph_pool_name; ceph_cluster_name = args.ceph_cluster_name
    ceph_user_name = args.ceph_user_name; ceph_conf_path = args.ceph_conf_path
    local_path = args.local_path; paired = args.paired

    record_batch = build_queues(key, parallel_enqueue=parallel_enqueue)

    ceph_params = {}
    ceph_params["user_name"] = ceph_user_name
    ceph_params["ceph_conf_path"] = ceph_conf_path
    ceph_params["cluster_name"] = ceph_cluster_name
    ceph_params["pool_name"] = ceph_pool_name


    pp = persona_ops.buffer_pool(size=1, bound=False)
    if local_path is None:
        parsed_chunks = tf.contrib.persona.persona_ceph_in_pipe(columns=["base", "qual"], ceph_params=ceph_params, 
                                    keys=record_batch, buffer_pool=pp, parse_parallel=parallel_dequeue, process_parallel=1)
    else:
        if len(record_batch) > 1:
          raise Exception("Local disk requires read parallelism of 1 (parallel_enqueue = 1)")
        mmap_pool = persona_ops.m_map_pool(size=10, bound=False, name="file_mmap_buffer_pool")
        parsed_chunks = tf.contrib.persona.persona_in_pipe(dataset_dir=local_path, columns=["base", "qual"], key=record_batch[0],
                                                           mmap_pool=mmap_pool, buffer_pool=pp, parse_parallel=parallel_dequeue, process_parallel=1)

    aligned_results, genome, options = create_ops(processed_batch=parsed_chunks, deep_verify=args.deep_verify,
                                                  num_aligners=args.aligners, num_writers=args.writers, subchunk_size=args.subchunking,
                                                  aligner_threads=args.aligner_threads, index_path=args.index_path, null_align=args.null,
                                                  paired=paired, max_secondary=args.max_secondary)
    genomes.append(genome)

    if args.writers == 0:
        results_out, key_out, _, _, = aligned_results[0]
        sink_op = persona_ops.buffer_list_sink(results_out)
        ops = sink_op
    elif local_path is None:
        # TODO the record name and the pool name are hardcoded for a single run. Need to restart to do different pools!
        columns = ['results']
        for i in range(args.max_secondary):
          columns.append("secondary{}".format(i))
        ops = tf.contrib.persona.persona_parallel_ceph_out_pipe(metadata_path=args.dataset, column=columns, 
                                            write_list_list=aligned_results, record_id="persona_results", compress=args.compress,
                                            ceph_params=ceph_params) 
        #ops = ceph_write_results(aligned_results=aligned_results, record_name=ceph_pool_name, compress_output=args.compress,
        #                         ceph_conf_path=ceph_conf_path, cluster_name=ceph_cluster_name, pool_name=ceph_pool_name,
        #                         user_name=ceph_user_name)
    else:
        #ops = local_write_results(aligned_results=aligned_results, output_path=local_path, record_name="bioflow_exp", compress_output=args.compress)
        columns = ['results']
        for i in range(args.max_secondary):
          columns.append("secondary{}".format(i))
        ops = tf.contrib.persona.persona_parallel_out_pipe(path=local_path, column=columns, 
                                            write_list_list=aligned_results, record_id="persona_results", compress=args.compress) 


    return [ops], genomes

    #run_aligner(sink_op=sink_op, genomes=genomes, summary=summary, null=True if args.null else False)

