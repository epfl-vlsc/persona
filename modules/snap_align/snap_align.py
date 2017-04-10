from __future__ import print_function

import os
import multiprocessing
from ..common.service import Service
from ..common.parse import numeric_min_checker, path_exists_checker, non_empty_string_checker

import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import queues, pipeline

class SnapCommonService(Service):
    columns = ["base", "qual"]

    def extract_run_args(self, args):
        dataset = args.dataset
        return (a["path"] for a in dataset["records"])

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

    def make_central_pipeline(self, args, input_gen, pass_around_gen):
        ready_to_process = pipeline.join(upstream_tensors=zip(input_gen, pass_around_gen),
                                         parallel=args.parallel,
                                         capacity=args.mmap_queue,
                                         multi=True)
        to_agd_reader, pass_around_agd_reader = zip(*ready_to_process)
        multi_column_gen = pipeline.agd_reader_multi_column_pipeline(upstream_tensorz=to_agd_reader)
        processed_bufs = tuple((processed_column, pass_around) for processed_column, pass_around in zip(multi_column_gen, pass_around_agd_reader))
        ready_to_assemble = pipeline.join(upstream_tensors=processed_bufs,
                                          parallel=4, capacity=32, multi=True) # TODO these params are kinda arbitrary :/
        # ready_to_assemble: [(output_buffers, num_records, first_ordinal, record_id), (pass_around)) x N]
        to_assembler, pass_around_assembler = zip(*ready_to_assemble)
        # each item out of this is a handle to AGDReads
        agd_read_assembler_gen = pipeline.agd_read_assembler(upstream_tensors=(a[:2] for a in to_assembler), include_meta=False)
        # assembled_records, ready_to_align: [(agd_reads_handle, (num_records, first_ordinal, record_id), (pass_around)) x N]
        assembled_records = zip(agd_read_assembler_gen, (a[1:] for a in to_assembler), pass_around_assembler)
        ready_to_align = pipeline.join(upstream_tensors=assembled_records,
                                       parallel=args.aligners, capacity=32, multi=True) # TODO still have default capacity here :/


        if args.paired:
            aligner_type = persona_ops.snap_align_paired
            aligner_options = persona_ops.paired_aligner_options(cmd_line="-o output.sam", name="paired_aligner_options")
        else:
            aligner_type = persona_ops.snap_align_single
            aligner_options = persona_ops.aligner_options(cmd_line="-o output.sam", name="aligner_options") # -o output.sam will not actually do anything

        aligner_dtype = tf.dtypes.string
        aligner_shape = tf.tensor_shape.matrix(rows=args.max_secondary+1, cols=2)
        first_assembled_result = ready_to_align[0][1:]
        sink_queue_shapes = [a.get_shape() for a in first_assembled_result]
        sink_queue_dtypes = [a.dtype for a in first_assembled_result]
        sink_queue_shapes.append(aligner_shape)
        sink_queue_dtypes.append(aligner_dtype)

        pass_around_aligners = (a[1:] for a in ready_to_align) # type: [((num_records, first_ordinal, record_id), (pass_around)) x N]
        pass_to_aligners = (a[0] for a in ready_to_align)

        buffer_list_pool = persona_ops.buffer_list_pool(**pipeline.pool_default_args) # TODO should this be passed in as argument?
        genome = persona_ops.genome_index(genome_location=args.index_path, name="genome_loader")

        def make_aligners(downstream_capacity=8):
            for read_handle, pass_around in zip(pass_to_aligners, pass_around_aligners):
                single_aligner_queue = tf.FIFOQueue(capacity=downstream_capacity,
                                                    dtypes=(aligner_dtype,),
                                                    shapes=(aligner_shape,),
                                                    name="aligner_post_queue")
                pass_around_sink = tf.train.batch(tensors=pass_around, batch_size=1)
                aligner_results = aligner_type(genome_handle=genome,
                                               options_handle=aligner_options,
                                               output_buffer_list_queue_handle=single_aligner_queue.queue_ref,
                                               num_threads=args.aligner_threads,
                                               read=read_handle,
                                               buffer_list_pool=buffer_list_pool,
                                               subchunk_size=args.subchunking,
                                               max_secondary=args.max_secondary)
                tf.train.queue_runner.add_queue_runner(tf.train.QueueRunner(queue=single_aligner_queue, enqueue_ops=(aligner_results,)))
                pass_around_sink.insert(0, single_aligner_queue.dequeue()) # returns buffer_list_result + pass_around
                yield pass_around_sink

        aligned_results = pipeline.join(upstream_tensors=make_aligners(), parallel=args.writers, capacity=32)
        return aligned_results # returns [(buffer_list_handle, (num_records, first_ordinal, record_id), (pass_around)) x N]


class CephCommonService(SnapCommonService):

    def input_dtypes(self):
        """ Ceph services require the key and the pool name """
        return [tf.string] * 2

    def input_shapes(self):
        """ Ceph services require the key and the pool name """
        return [tf.TensorShape([])] * 2

    def extract_run_args(self, args):
        dataset = args.dataset
        pool_key = "ceph_pool" # TODO what is it actually?
        if pool_key not in dataset:
            raise Exception("key '{k}' not found in dataset keys {keys}".format(k=pool_key, keys=dataset.keys()))
        ceph_pool = dataset[pool_key] # TODO might require fancier extraction
        return ((a, ceph_pool) for a in super().extract_run_args(args=args))

    def add_graph_args(self, parser):
        super().add_graph_args(parser=parser)
        parser.add_argument("--ceph-cluster-name", type=non_empty_string_checker, default="ceph", help="name for the ceph cluster")
        parser.add_argument("--ceph-user-name", type=non_empty_string_checker, default="client.dcsl1024", help="ceph username")
        parser.add_argument("--ceph-conf-path", type=path_exists_checker(check_dir=False), default="/etc/ceph/ceph.conf", help="path for the ceph configuration")
        parser.add_argument("--ceph-read-chunk-size", default=(2**26), type=numeric_min_checker(128, "must have a reasonably large minimum read size from Ceph"), help="minimum size to read from ceph storage, in bytes")

class CephSnapService(CephCommonService):
    """ A service to use the snap aligner with a ceph dataset """

    def get_shortname(self):
        return "ceph"

    def output_dtypes(self):
        return ((tf.dtypes.string,) * 3) + (tf.dtypes.int32, tf.dtypes.int64, tf.dtypes.string)

    def output_shapes(self):
        return (tf.tensor_shape.scalar(),) * 6

    def make_graph(self, in_queue, args):
        """
        :param in_queue: 
        :param args: 
        :return: [ key, pool_name, num_records, first_ordinal, record_id, full_path ]
        """
        parallel_key_dequeue = (in_queue.dequeue() for _ in range(args.enqueue))
        # parallel_key_dequeue = [(key, pool_name) x N]
        ceph_read_buffers = tuple(pipeline.ceph_read_pipeline(upstream_tensors=parallel_key_dequeue,
                                                              user_name=args.ceph_user_name,
                                                              cluster_name=args.ceph_cluster_name,
                                                              ceph_conf_path=args.ceph_conf_path,
                                                              columns=self.columns))
        pass_around_central_gen = (a[:2] for a in ceph_read_buffers) # key, pool_name
        to_central_gen = (a[2] for a in ceph_read_buffers) # [ chunk_buffer_handles x N ]

        # aligner_results: [(buffer_list_result, (num_records, first_ordinal, record_id), (key, pool_name)) x N]
        aligner_results = self.make_central_pipeline(args=args,
                                                     input_gen=to_central_gen,
                                                     pass_around_gen=pass_around_central_gen)
        to_writer_gen = ((key, pool_name, num_records, first_ordinal, record_id, buffer_list_ref) for buffer_list_ref, (num_records, first_ordinal, record_id), (key, pool_name) in aligner_results)

        # ceph writer pipeline wants (key, first_ord, num_recs, pool_name, record_id, column_handle)

        # type of aligned_results_queue: [(key, pool_name, num_records, first_ordinal, record_id, buffer_list_ref) x N]
        # this happens to match the iterator in ceph_aligner_write_pipeline, but otherwise you can mix like above
        writer_outputs = pipeline.ceph_aligner_write_pipeline(
            upstream_tensors=to_writer_gen,
            user_name=args.ceph_user_name,
            cluster_name=args.ceph_cluster_name,
            ceph_conf_path=args.ceph_conf_path
        )
        return (b+(a,) for a,b in zip(writer_outputs, ((key, pool_name, num_records, first_ordinal, record_id)
                                                       for _, (num_records, first_ordinal, record_id), (key, pool_name) in aligner_results)))


class CephNullService(CephCommonService):
    """ A service to read and write from ceph as if we were a performing real alignment,
     but it performs no alignment """

    def output_dtypes(self):
        pass

    def output_shapes(self):
        pass

    def make_graph(self, in_queue, args):
        pass

class LocalCommonService(SnapCommonService):
    def extract_run_args(self, args):
        dataset_dir = args.dataset_dir
        return (os.path.join(dataset_dir, a) for a in super().extract_run_args(args=args))

    def add_run_args(self, parser):
        super().add_graph_args(parser=parser)
        parser.add_argument("-d", "--dataset-dir", type=path_exists_checker(), required=True, help="Directory containing ALL of the chunk files")

class LocalSnapService(LocalCommonService):
    """ A service to use the SNAP aligner with a local dataset """

    def get_shortname(self):
        return "local"

    def output_dtypes(self):
        return ((tf.dtypes.string,) * 2) + (tf.dtypes.int32, tf.dtypes.int64, tf.dtypes.string)

    def output_shapes(self):
        return (tf.tensor_shape.scalar(),) * 5

    def make_graph(self, in_queue, args):
        parallel_key_dequeue = tuple(in_queue.dequeue() for _ in range(args.enqueue))
        # read_files: [(file_path, (mmaped_file_handles, a gen)) x N]
        read_files = pipeline.local_read_pipeline(upstream_tensors=parallel_key_dequeue, columns=self.columns)
        # need to use tf.tuple to make sure that these are both made ready at the same time
        combo = tuple((tf.tuple((file_basename,) + tuple(read_handle_gen) for file_basename, read_handle_gen in zip(parallel_key_dequeue, read_files))))
        to_central_gen = (a[1:] for a in combo)
        pass_around_gen = (a[0] for a in combo)

        aligner_results = self.make_central_pipeline(args=args,
                                                     input_gen=to_central_gen,
                                                     pass_around_gen=pass_around_gen)

        to_writer_gen = ((buffer_list_handle, record_id, first_ordinal, num_records, file_basename) for buffer_list_handle, (num_records, first_ordinal, record_id), file_basename in aligner_results)
        written_records = pipeline.local_write_pipelien(upstream_tensors=to_writer_gen)
        final_output_gen = zip(written_records, ((record_id, first_ordinal, num_records, file_basename) for _, (num_records, first_ordinal, record_id), file_basename in aligner_results))
        return (b+(a,) for a,b in final_output_gen)


class LocalNullService(LocalCommonService):
    """ A service to read and write from a local dataset as if we were a performing real alignment,
     but it performs no alignment """

    def output_dtypes(self):
        pass

    def output_shapes(self):
        pass

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

