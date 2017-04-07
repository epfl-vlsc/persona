from __future__ import print_function

import multiprocessing
from ..common.service import Service
from ..common.parse import numeric_min_checker, path_exists_checker, non_empty_string_checker

import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()
from tf.contrib.persona import queues, pipeline

class SnapCommonService(Service):
    columns = ["base", "qual"]

    def tooltip(self):
        return self.__doc__

    def add_graph_args(self, parser):
        # adds the common args to all graphs
        parser.add_argument("-p", "--parallel", type=numeric_min_checker(1, "parallel decompression"), default=2, help="parallel decompression")
        parser.add_argument("-e", "--enqueue", type=numeric_min_checker(1, "parallel enqueuing"), default=1, help="parallel enqueuing / reading from Ceph")
        parser.add_argument("-m", "--mmap-queue", type=numeric_min_checker(1, "mmap queue size"), default=2, help="size of the mmaped file record queue")
        parser.add_argument("-a", "--aligners", type=numeric_min_checker(1, "number of aligners"), default=1, help="number of aligners")
        parser.add_argument("-t", "--aligner-threads", type=numeric_min_checker(1, "threads per aligner"), default=multiprocessing.cpu_count(), help="the number of threads to use per aligner")
        parser.add_argument("-x", "--subchunking", type=numeric_min_checker(100, "don't go lower than 100 for subchunking size"), default=5000, help="the size of each subchunk (in number of reads)")
        parser.add_argument("-w", "--writers", type=numeric_min_checker(0, "must have a non-negative number of writers"), default=0, help="the number of writer pipelines")
        #parser.add_argument("-c", "--compress", default=False, action='store_true', help="compress the output")
        parser.add_argument("-s", "--max-secondary", type=numeric_min_checker(0, "must have a non-negative number of secondary results"), default=0, help="Max secondary results to store. >= 0 ")
        parser.add_argument("--deep-verify", default=False, action='store_true', help="verify record integrity")
        parser.add_argument("--paired", default=False, action='store_true', help="interpret dataset as interleaved paired dataset")
        parser.add_argument("--summary", default=False, action="store_true", help="Add TensorFlow summary info to the graph")
        parser.add_argument("-i", "--index-path", type=path_exists_checker(), default="/scratch/stuart/ref_index", help="location of the ref index on all machines. Make sure all machines have this path!")

class CephCommonService(SnapCommonService):

    def input_dtypes(self):
        """ Ceph services require the key and the pool name """
        return [tf.string] * 2

    def input_shapes(self):
        """ Ceph services require the key and the pool name """
        return [tf.TensorShape([])] * 2

    def add_graph_args(self, parser):
        super().add_graph_args(parser=parser)
        parser.add_argument("--ceph-cluster-name", type=non_empty_string_checker, default="ceph", help="name for the ceph cluster")
        parser.add_argument("--ceph-user-name", type=non_empty_string_checker, default="client.dcsl1024", help="ceph username")
        parser.add_argument("--ceph-conf-path", type=path_exists_checker(check_dir=False), default="/etc/ceph/ceph.conf", help="path for the ceph configuration")
        parser.add_argument("--ceph-read-chunk-size", default=(2**26), type=numeric_min_checker(128, "must have a reasonably large minimum read size from Ceph"), help="minimum size to read from ceph storage, in bytes")

class CephSnapService(CephCommonService):
    """ A service to use the snap aligner with a ceph dataset """
    def output_dtypes(self):
        return ((tf.dtypes.string) * 3) + (tf.dtypes.int32, tf.dtypes.int64, tf.dtypes.string)

    def output_shapes(self):
        return (tf.tensor_shape.scalar(),) * 6

    def make_graph(self, in_queue, args):
        parallel_key_dequeue = (in_queue.dequeue() for _ in range(args.enqueue))
        # parallel_key_dequeue = [(key, pool_name) x N]
        ceph_read_buffers = pipeline.ceph_read_pipeline(upstream_tensors=parallel_key_dequeue,
                                                        user_name=args.ceph_user_name,
                                                        cluster_name=args.ceph_cluster_name,
                                                        ceph_conf_path=args.ceph_conf_path,
                                                        columns=self.columns)
        # ceph_read_buffers = [(key, pool_name, chunk_buffers) x N]
        ready_to_process_bufs = pipeline.join(upstream_tensors=ceph_read_buffers,
                                              parallel=args.parallel,
                                              capacity=args.mmap_queue,
                                              multi=True)

        to_agd_reader = (a[2] for a in ready_to_process_bufs)
        pass_around_agd_reader = (a[:2] for a in ready_to_process_bufs)
        multi_column_gen = pipeline.agd_reader_multi_column_pipeline(upstream_tensorz=to_agd_reader)
        processed_bufs = tuple(pass_around + processed_tuple for pass_around, processed_tuple in zip(pass_around_agd_reader, multi_column_gen))
        assert len(processed_bufs) == len(ready_to_process_bufs) # TODO get rid of this assert and make the above statement back into a generator

        ready_to_assemble = pipeline.join(upstream_tensors=processed_bufs,
                                          parallel=4, capacity=32, multi=True) # Just made up these params :/

        # ready_to_assemble: [(key, pool_name, output_buffers, num_records, first_ordinal, record_id) x N]
        to_agd_assembler = ((output_buffers, num_records) for _, _, output_buffers, num_records, _, _ in ready_to_assemble)
        pass_around_agd_assembler = ((key, pool_name, num_records, first_ordinal, record_id) for key, pool_name, _, num_records, first_ordinal, record_id in ready_to_assemble)
        agd_read_assembler_gen = pipeline.agd_read_assembler(upstream_tensors=to_agd_assembler, include_meta=False)

        assembled_records = tuple(pass_around + processed_tuple for pass_around, processed_tuple in zip(pass_around_agd_assembler, agd_read_assembler_gen))
        assert len(assembled_records) == len(ready_to_assemble)

        ready_to_align = pipeline.join(upstream_tensors=assembled_records,
                                       parallel=args.aligners, capacity=32, multi=True) # TODO still have default args here

        # TODO move all of this stuff into a help method, as it should be common for all

        if args.paired:
            aligner_type = persona_ops.snap_align_paired
            aligner_options = persona_ops.paired_aligner_options(cmd_line="-o output.sam", name="paired_aligner_options")
        else:
            aligner_type = persona_ops.snap_align_single
            aligner_options = persona_ops.aligner_options(cmd_line="-o output.sam", name="aligner_options") # -o output.sam will not actually do anything

        aligner_dtype = tf.dtypes.string
        aligner_shape = tf.tensor_shape.matrix(rows=args.max_secondary+1, cols=2)

        def make_aligners(pass_around_gen, read_handles_gen, genome, options, buffer_list_pool, downstream_capacity=8): # TODO not sure what this capacity should be
            for read_handle, pass_around in zip(read_handles_gen, pass_around_gen):
                single_aligner_queue = tf.FIFOQueue(capacity=downstream_capacity,
                                                    dtypes=(aligner_dtype,),
                                                    shapes=(aligner_shape,),
                                                    name="aligner_post_queue")
                pass_around_sink = tf.train.batch(tensors=pass_around, batch_size=1)
                aligner_results = aligner_type(genome_handle=genome,
                                               options_handle=options,
                                               output_buffer_list_queue_handle=single_aligner_queue.queue_ref,
                                               num_threads=args.aligner_threads,
                                               read=read_handle,
                                               buffer_list_pool=buffer_list_pool,
                                               subchunk_size=args.subchunking,
                                               max_secondary=args.max_secondary)
                tf.train.queue_runner.add_queue_runner(tf.train.QueueRunner(queue=single_aligner_queue, enqueue_ops=(aligner_results,)))
                pass_around_sink.append(single_aligner_queue.dequeue())
                yield pass_around_sink

        # ready_to_align: [(key, pool_name, num_records, first_ordinal, record_id, agd_read_handle) x N]
        pass_around_example = ready_to_align[0][:-1]
        aligner_sink_queue_shape = [a.get_shape() for a in pass_around_example]
        aligner_sink_queue_data_type = [a.dtype for a in pass_around_example]
        aligner_sink_queue_shape.append(aligner_dtype)
        aligner_sink_queue_data_type.append(aligner_shape)

        buffer_list_pool = persona_ops.buffer_list_pool(**pipeline.pool_default_args)

        genome = persona_ops.genome_index(genome_location=args.index_path, name="genome_loader")
        pass_around_gen = (a[:-1] for a in ready_to_align)
        agd_read_handles_gen = (a[-1] for a in ready_to_align)

        aligner_ready_results = pipeline.join(upstream_tensors=make_aligners(
            pass_around_gen=pass_around_gen, read_handles_gen=agd_read_handles_gen, genome=genome,
            options=aligner_options, buffer_list_pool=buffer_list_pool
        ), parallel=args.writers, capacity=32) # not sure what to do for this

        # type of aligned_results_queue: [(key, pool_name, num_records, first_ordinal, record_id, buffer_list_ref) x N]
        # this happens to match the iterator in ceph_aligner_write_pipeline, but otherwise you can mix like above
        writer_outputs = pipeline.ceph_aligner_write_pipeline(
            upstream_tensors=aligner_ready_results,
            user_name=args.ceph_user_name,
            cluster_name=args.ceph_cluster_name,
            ceph_conf_path=args.ceph_conf_path
        )
        final_params_gen = (a[:-1] for a in aligner_ready_results)

        # each item is [(final_write_key_with_extension, key, pool_name, num_records, first_ordinal, record_id) x N]
        return tuple((writer_output,)+final_params for final_params, writer_output in zip(final_params_gen, writer_outputs))


class CephNullService(CephCommonService):
    """ A service to read and write from ceph as if we were a performing real alignment,
     but it performs no alignment """

    def output_dtypes(self):
        pass

    def output_shapes(self):
        pass

    def make_graph(self, in_queue, args):
        pass

class LocalSnapService(SnapCommonService):
    """ A service to use the SNAP aligner with a local dataset """

    def output_dtypes(self):
        pass

    def output_shapes(self):
        pass

class LocalNullService(SnapCommonService):
    """ A service to read and write from a local dataset as if we were a performing real alignment,
     but it performs no alignment """

    def output_dtypes(self):
        pass

    def output_shapes(self):
        pass

class CephSnapService(Service):

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
        options = persona_ops.paired_aligner_options(cmd_line="-o output.sam", name="paired_aligner_options")
    else:
        options = persona_ops.aligner_options(cmd_line="-o output.sam", name="aligner_options") # -o output.sam will not actually do anything

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

