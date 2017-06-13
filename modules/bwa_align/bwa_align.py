#!/usr/bin/env python3
from __future__ import print_function

import multiprocessing
import os
import shutil
import json
from ..common.service import Service
from common.parse import numeric_min_checker, path_exists_checker, non_empty_string_checker
import common.parse as parse

import tensorflow as tf
import itertools

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import queues, pipeline

class BWACommonService(Service):
    columns = ["base", "qual"]
    write_columns = []

    def extract_run_args(self, args):
        if args.paired and args.aligner_threads < 3:
            raise Exception("Need at least 3 aligner threads for paired execution")
        dataset = args.dataset
        return (a["path"] for a in dataset["records"])

    def add_graph_args(self, parser):
        # adds the common args to all graphs
        parser.add_argument("-p", "--parallel", type=int, default=2, help="parallel decompression")
        parser.add_argument("-e", "--enqueue", type=int, default=1, help="parallel enqueuing")
        parser.add_argument("-m", "--mmap-queue", type=int, default=2, help="size of the mmaped file record queue")
        parser.add_argument("-a", "--aligners", type=numeric_min_checker(1, "number of aligners"), default=1, help="number of aligners")
        parser.add_argument("-t", "--aligner-threads", type=numeric_min_checker(1, "number of aligner threads"), 
            default=multiprocessing.cpu_count(), help="the number of threads to use for alignment. >= 1 or >= 3 if paired [num_cpus]")
        parser.add_argument("-r", "--thread-ratio", type=float, default=0.66, help="Ratio of aligner threads to finalize threads")
        parser.add_argument("-x", "--subchunking", type=int, default=5000, help="the size of each subchunk (in number of reads)")
        parser.add_argument("-w", "--writers", type=int, default=0, help="the number of writer pipelines")
        parser.add_argument("-c", "--compress", default=False, action='store_true', help="compress the output")
        parser.add_argument("-i", "--index-path", default="/scratch/bwa_index/hs38DH.fa")
        parser.add_argument("-s", "--max-secondary", default=1, help="Max secondary results to store. >= 1 (required for chimaric results")
        parser.add_argument("--paired", default=False, action='store_true', help="interpret dataset as interleaved paired dataset")
        parser.add_argument("--null", type=float, required=False, help="use the null aligner instead of actually aligning")
        parser.add_argument("--deep-verify", default=False, action='store_true', help="verify record integrity")
        # TODO this is rigid, needs to be changed to get from the queue service!
        parser.add_argument("--bwa-args", default="", help="BWA algorithm options")

    def make_central_pipeline(self, args, input_gen, pass_around_gen):
        
        self.write_columns.append('results')
        for i in range(args.max_secondary):
            self.write_columns.append('secondary{}'.format(i))

        self.write_columns = [ {"type": "structured", "extension": a} for a in self.write_columns]
        
        joiner = tuple(tuple(a) + tuple(b) for a,b in zip(input_gen, pass_around_gen))
        ready_to_process = pipeline.join(upstream_tensors=joiner,
                                         parallel=args.parallel,
                                         capacity=args.mmap_queue,
                                         multi=True)
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
                                          parallel=1, capacity=32, multi=True) # TODO these params are kinda arbitrary :/
        # ready_to_assemble: [output_buffers, num_records, first_ordinal, record_id, pass_around {flattened}) x N]
        to_assembler, pass_around_assembler = zip(*((a[:2], a[1:]) for a in ready_to_assemble))
        #print("reads {}".format(to_assembler))
        base, qual = tf.unstack(to_assembler[0][0])
        num_recs = to_assembler[0][1]
        two_bit_base = persona_ops.two_bit_converter(num_recs, base)
        to_assembler_converted = [[ tf.stack([two_bit_base, qual]) , num_recs ]]
        # each item out of this is a handle to AGDReads
        agd_read_assembler_gen = tuple(pipeline.agd_bwa_read_assembler(upstream_tensors=to_assembler_converted, include_meta=False))
        # assembled_records, ready_to_align: [(agd_reads_handle, (num_records, first_ordinal, record_id), (pass_around)) x N]
        assembled_records_gen = tuple(zip(agd_read_assembler_gen, pass_around_assembler))
        assembled_records = tuple(((a,) + tuple(b)) for a,b in assembled_records_gen)
        ready_to_align = pipeline.join(upstream_tensors=assembled_records,
                                       parallel=args.aligners, capacity=32, multi=True) # TODO still have default capacity here :/

        options = persona_ops.bwa_options(options=args.bwa_args.split(), name="bwa_aligner_options")
        bwa_index = persona_ops.bwa_index(index_location=args.index_path, ignore_alt=False, name="index_loader")

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

        if args.paired:
            executor = persona_ops.bwa_paired_executor(max_secondary=args.max_secondary,
                                                       num_threads=args.aligner_threads,
                                                       work_queue_size=args.aligners+1,
                                                       options_handle=options,
                                                       index_handle=bwa_index,
                                                       thread_ratio=0.66)
        else:
            executor = persona_ops.bwa_single_executor(max_secondary=args.max_secondary,
                                                       num_threads=args.aligner_threads,
                                                       work_queue_size=args.aligners+1,
                                                       options_handle=options,
                                                       index_handle=bwa_index)
        def make_aligners():

            aligner_op = persona_ops.bwa_align_single if not args.paired else persona_ops.bwa_align_paired

            for read_handle, pass_around in zip(pass_to_aligners, pass_around_aligners):
                aligner_results = aligner_op(read=read_handle,
                                               buffer_list_pool=buffer_list_pool,
                                               subchunk_size=args.subchunking,
                                               executor_handle=executor,
                                               max_secondary=args.max_secondary)
                #print(aligner_results)
                yield (aligner_results,) + tuple(pass_around)

        aligners = tuple(make_aligners())
        aligned_results = pipeline.join(upstream_tensors=aligners, parallel=args.writers, multi=True, capacity=32)
        
        ref_seqs, lens = persona_ops.bwa_index_reference_sequences(index_handle=bwa_index)
        return aligned_results,  (bwa_index, ref_seqs, lens) # returns [(buffer_list_handle, num_records, first_ordinal, record_id, pass_around X N) x N], that is COMPLETELY FLAT

class CephCommonService(BWACommonService):

    def input_dtypes(self, args):
        """ Ceph services require the key and the pool name """
        return [tf.string] * 2

    def input_shapes(self, args):
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

class CephBWAService(CephCommonService):
    """ A service to use the bwa aligner with a ceph dataset """

    def get_shortname(self):
        return "ceph"

    def output_dtypes(self, args):
        return ((tf.dtypes.string,) * 3) + (tf.dtypes.int32, tf.dtypes.int64, tf.dtypes.string)

    def output_shapes(self, args):
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

class LocalCommonService(BWACommonService):
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

class LocalBWAService(LocalCommonService):
    """ A service to use the BWA aligner with a local dataset """

    def get_shortname(self):
        return "local"

    def output_dtypes(self, args):
        return ((tf.dtypes.string,) * 2) + (tf.dtypes.int32, tf.dtypes.int64, tf.dtypes.string)

    def output_shapes(self, args):
        return (tf.tensor_shape.scalar(),) * 5

    def make_graph(self, in_queue, args):
        parallel_key_dequeue = tuple(in_queue.dequeue() for _ in range(args.enqueue))
        # read_files: [(file_path, (mmaped_file_handles, a gen)) x N]
        read_files = tuple(tf.tuple((path_base,) + tuple(read_gen)) for path_base, read_gen in zip(parallel_key_dequeue, pipeline.local_read_pipeline(upstream_tensors=parallel_key_dequeue, columns=self.columns)))
        # need to use tf.tuple to make sure that these are both made ready at the same time
        to_central_gen = tuple(a[1:] for a in read_files)
        pass_around_gen = tuple((a[0],) for a in read_files)

        aligner_results, run_first = tuple(self.make_central_pipeline(args=args,
                                                                      input_gen=to_central_gen,
                                                                      pass_around_gen=pass_around_gen))

        to_writer_gen = tuple((buffer_list_handle, record_id, first_ordinal, num_records, file_basename) for buffer_list_handle, num_records, first_ordinal, record_id, file_basename in aligner_results)
        #print(to_writer_gen)
        #print(self.write_columns)
        written_records = tuple(tuple(a) for a in pipeline.local_write_pipeline(upstream_tensors=to_writer_gen, record_types=self.write_columns, compressed=False))
        final_output_gen = zip(written_records, ((record_id, first_ordinal, num_records, file_basename) for _, num_records, first_ordinal, record_id, file_basename in aligner_results))
        return (b+(a,) for a,b in final_output_gen), run_first


# old stuff



class BwaService(Service):
    """ A class representing a service module in Persona """
   
    #default inputs

    def output_dtypes(self, args):
        return []
    def output_shapes(self, args):
        return []
    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two 
        things: a list of tensors which the runtime will 
        evaluate, and a list of run-once ops"""
        # make the graph
        if args.null is not None:
          if args.null < 0.0:
              raise EnvironmentError("null wait time must be strictly non-negative. Got {}".format(args.null))

        index_path = args.index_path
        if not (os.path.exists(index_path) and os.path.isfile(index_path)):
          raise EnvironmentError("Index path '{}' specified incorrectly. Should be path/to/index.fa".format(index_path))

        #print("aligner {} finalizer {}".format(args.aligner_threads, args.finalizer_threads))

        if args.writers < 0:
          raise EnvironmentError("need a strictly positive number of writers, got {}".format(args.writers))
        if args.parallel < 1:
          raise EnvironmentError("need at least 1 parallel dequeue, got {}".format(args.parallel))
        if args.enqueue < 1:
          raise EnvironmentError("need at least 1 parallel enqueue, got {}".format(args.enqueue))

        if in_queue is None:
          if not os.path.isfile(args.dataset):
            raise EnvironmentError("Not a valid dataset: {}".format(args.dataset))
          else:
            raise EnvironmentError("in queue was None")

        key = in_queue.dequeue()
        ops, run_once = run(key, args)

        return ops, run_once

bwa_service_ = BwaService()

def service():
    return bwa_service_

def create_ops(key, deep_verify, aligner_threads, finalizer_threads, subchunk_size, num_writers, index_path, null_align, paired, dataset_dir, max_secondary):

    bwa_index = persona_ops.bwa_index(index_location=index_path, ignore_alt=False, name="index_loader")
    option_line = [e[1:] for e in "-M".split()]
    options = persona_ops.bwa_options(options=option_line, name="paired_aligner_options")

    pp = persona_ops.buffer_pool(size=1, bound=False)
    drp = persona_ops.bwa_read_pool(size=0, bound=False)
    mmap_pool = persona_ops.m_map_pool(size=10, bound=False, name="file_mmap_buffer_pool")

    parsed_chunks = tf.contrib.persona.persona_in_pipe(dataset_dir=dataset_dir, columns=["base", "qual"], key=key, 
                                                       mmap_pool=mmap_pool, buffer_pool=pp)
    aggregate_enqueue = []
    for chunk in parsed_chunks:

        key = chunk[0]
        num_reads = chunk[1]
        first_ord = chunk[2]
        base_reads = chunk[3]
        qual_reads = chunk[4]

        agd_read = persona_ops.no_meta_bwa_assembler(bwa_read_pool=drp,
                                                  base_handle=base_reads,
                                                  qual_handle=qual_reads,
                                                  num_records=num_reads, name="bwa_assembler")

        aggregate_enqueue.append((agd_read, key, num_reads, first_ord))

    record, key, num_records, first_ordinal = tf.contrib.persona.batch_join_pdq(tensor_list_list=[e for e in aggregate_enqueue],
                                                    batch_size=1, capacity=8,
                                                    enqueue_many=False,
                                                    num_dq_ops=1,
                                                    name="agd_reads_to_aligners")[0]
    
    blp = persona_ops.buffer_list_pool(size=1, bound=False)
   
    if paired:
        # all this is a little dirty because we need to pass key, num_recs and ordinal through the pipe as well
        # first stage align
        aligned_regs = persona_ops.bwa_aligner(index_handle=bwa_index, options_handle=options, num_threads=aligner_threads,
                                      max_read_size=400, read=record, subchunk_size=subchunk_size, name="bwa_align")

        regs_out = tf.contrib.persona.batch_pdq(tensor_list=[aligned_regs, key, num_records, first_ordinal], batch_size=1, enqueue_many=False,
                                      num_dq_ops=1, name="align_to_pestat_queue")[0]

        # second stage infer insert size for chunk
        regs_out_us = regs_out[0]
        regs_stat = persona_ops.bwa_paired_end_stat(index_handle=bwa_index, options_handle=options, 
                                                    read=regs_out_us, name="bwa_pe_stat");

        to_enq = [regs_stat, regs_out[1], regs_out[2], regs_out[3]]
        regs_stat_out = tf.contrib.persona.batch_pdq(tensor_list=to_enq, 
                                      batch_size=1, enqueue_many=False,
                                      num_dq_ops=1, name="pestat_to_finalizer_queue")[0]
       
        #final stage, pair selection
        regs_stat_us = regs_stat_out[0]
        aligned_result = persona_ops.bwa_finalize(index_handle=bwa_index, options_handle=options, num_threads=finalizer_threads,
                                      buffer_list_pool=blp, read=regs_stat_us, subchunk_size=subchunk_size, 
                                      max_read_size=400, name="bwa_finalize")
        
        aligned_results = tf.contrib.persona.batch_pdq(tensor_list=[aligned_result, regs_stat_out[1], regs_stat_out[2], regs_stat_out[3]],
                                                batch_size=1, capacity=8, #this queue should always be empty!
                                                num_dq_ops=max(1, num_writers), name="results_to_sink")
    else:
        aligned = persona_ops.bwa_align_single(index_handle=bwa_index, options_handle=options, num_threads=aligner_threads,
                                      max_read_size=400, read=record, subchunk_size=subchunk_size, buffer_list_pool=blp, max_secondary=max_secondary, name="bwa_align")
        aligned_results = tf.contrib.persona.batch_pdq(tensor_list=[aligned, key, num_records, first_ordinal], batch_size=1, enqueue_many=False,
                                      num_dq_ops=1, name="align_results_out")

    return aligned_results, bwa_index, options


def run_aligner(final_op, indexes, summary, null, metadata_path, max_secondary):
    #trace_dir = setup_output_dir()
    init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    merged = tf.summary.merge_all()

    if not isinstance(final_op, (list, tuple)):
      ops = [final_op]
    else:
      ops = final_op

    index_refs = [persona_ops.bwa_index_reference_sequences(indexes[0])]
    
    print(os.getpid())

    with tf.Session() as sess:
        if summary:
            trace_dir = setup_output_dir()
            ops.append(merged)
            summary_writer = tf.summary.FileWriter("{trace_dir}/tf_summary".format(trace_dir=trace_dir), graph=sess.graph, max_queue=2**20, flush_secs=10**4)
            count = 0
        sess.run(init_ops)  # required for epoch input variable
        if not null:
            sess.run(indexes)

        coord = tf.train.Coordinator()
        print("Starting Run")
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        while not coord.should_stop():
            try:
                a = sess.run(ops)
                if summary:
                    summary_writer.add_summary(a[-1], global_step=count)
                    count += 1
            except tf.errors.OutOfRangeError as e:
                print('Got out of range error!')
                break
        print("Coord requesting stop")
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
      
        # now, add the reference info to the metadata file
        print("adding ref data at {}".format(metadata_path))
        refs_out = sess.run(index_refs)[0]
        refs = refs_out.ref_seqs
        lens = refs_out.ref_lens
        with open(metadata_path, 'r') as j:
            metadata = json.load(j)
        ref_list = []
        for i, ref in enumerate(refs):
          ref_list.append((ref.decode("utf-8"), lens[i].item()))
        metadata['reference_contigs'] = ref_list
        metadata['reference'] = index_path
        # adjust columns field with number of secondary results
        columns = ['base', 'qual', 'meta', 'results']
        for i in range(max_secondary):
          columns.append("secondary{}".format(i))
        metadata['columns'] = columns
        with open("test.json", 'w') as j:
          json.dump(obj=metadata, fp=j, indent=4)



def setup_output_dir(dirname="cluster_traces"):
    trace_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dirname)
    if os.path.exists(trace_path):
        # nuke it
        shutil.rmtree(trace_path)
    os.makedirs(trace_path)
    return trace_path



def run(key, args):
    all_ops = []
    indexes = []

    parallel_enqueue = args.enqueue; parallel_dequeue = args.parallel
    mmap_queue_length = args.mmap_queue; 
    paired = args.paired
    summary = args.summary
    dataset_dir = os.path.dirname(args.dataset)


    aligned_results, bwa_index, options = create_ops(key=key, deep_verify=args.deep_verify,
                                                  num_writers=args.writers, subchunk_size=args.subchunking,
                                                  aligner_threads=args.aligner_threads, finalizer_threads=args.finalizer_threads, 
                                                  index_path=args.index_path, null_align=args.null,
                                                  paired=paired, dataset_dir=dataset_dir, max_secondary=args.max_secondary)
    indexes.append(bwa_index)

    if args.writers == 0:
        results_out, key_out, _, _, = aligned_results[0]
        sink_op = persona_ops.buffer_list_sink(results_out)
        ops = sink_op
    else:
        columns = ['results']
        for i in range(args.max_secondary):
          columns.append("secondary{}".format(i))
        ops = tf.contrib.persona.persona_parallel_out_pipe(path=dataset_dir, column=columns, 
                                            write_list_list=aligned_results, record_id="persona_results", compress=args.compress) 

    all_ops = ops
    return all_ops, indexes

    #run_aligner(final_op=all_ops, indexes=indexes, summary=summary, null=True if args.null else False, metadata_path=args.metadata_file,
    #    max_secondary=args.max_secondary)

