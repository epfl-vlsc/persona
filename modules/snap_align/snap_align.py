from __future__ import print_function

import multiprocessing
from ..common.service import Service
from ..common.parse import numeric_min_checker, path_exists_checker, non_empty_string_checker

import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()

class SnapCommonService(Service):
    def tooltip(self):
        return self.__doc__

    def add_graph_args(self, parser):
        # adds the common args to all graphs
        parser.add_argument("-p", "--parallel", type=numeric_min_checker(1, "parallel decompression"), default=2, help="parallel decompression")
        parser.add_argument("-e", "--enqueue", type=numeric_min_checker(1, "parallel enqueuing"), default=1, help="parallel enqueuing")
        parser.add_argument("-m", "--mmap-queue", type=numeric_min_checker(1, "mmap queue size"), default=2, help="size of the mmaped file record queue")
        parser.add_argument("-a", "--aligners", type=numeric_min_checker(1, "number of aligners"), default=1, help="number of aligners")
        parser.add_argument("-t", "--aligner-threads", type=numeric_min_checker(1, "threads per aligner"), default=multiprocessing.cpu_count(), help="the number of threads to use per aligner")
        parser.add_argument("-x", "--subchunking", type=numeric_min_checker(100, "don't go lower than 100 for subchunking size"), default=5000, help="the size of each subchunk (in number of reads)")
        parser.add_argument("-w", "--writers", type=numeric_min_checker(0, "must have a non-negative number of writers"), default=0, help="the number of writer pipelines")
        parser.add_argument("-c", "--compress", default=False, action='store_true', help="compress the output")
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
        return [tf.TensorShape()] * 2

    def add_graph_args(self, parser):
        super().add_graph_args(parser=parser)
        parser.add_argument("--ceph-cluster-name", type=non_empty_string_checker, default="ceph", help="name for the ceph cluster")
        parser.add_argument("--ceph-user-name", type=non_empty_string_checker, default="client.dcsl1024", help="ceph username")
        parser.add_argument("--ceph-conf-path", type=path_exists_checker(check_dir=False), default="/etc/ceph/ceph.conf", help="path for the ceph configuration")
        parser.add_argument("--ceph-read-chunk-size", default=(2**26), type=numeric_min_checker(128, "must have a reasonably large minimum read size from Ceph"), help="minimum size to read from ceph storage, in bytes")

class CephSnapService(CephCommonService):
    """ A service to use the snap aligner with a ceph dataset """
    def output_dtypes(self):
        pass

    def output_shapes(self):
        pass

    def make_graph(self, in_queue, args):
        pass

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

