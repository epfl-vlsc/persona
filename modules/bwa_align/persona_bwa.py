#!/usr/bin/env python3
from __future__ import print_function

import argparse
import multiprocessing
import os
import shutil
import json
import time

import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()

def setup_output_dir(dirname="cluster_traces"):
    trace_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dirname)
    if os.path.exists(trace_path):
        # nuke it
        shutil.rmtree(trace_path)
    os.makedirs(trace_path)
    return trace_path


def get_metadata_attributes(json_file):
    with open(json_file, 'r') as j:
        metadata = json.load(j)
    records = metadata["records"]
    chunk_keys = list(a["path"] for a in records)
    print(chunk_keys)
    return chunk_keys

def create_ops(metadata_json, deep_verify, aligner_threads, finalizer_threads, subchunk_size, num_writers, index_path, null_align, paired, dataset_dir, max_secondary):

    bwa_index = persona_ops.bwa_index(index_location=index_path, ignore_alt=False, name="index_loader")
    option_line = [e[1:] for e in "-M".split()]
    options = persona_ops.bwa_options(options=option_line, name="paired_aligner_options")

    pp = persona_ops.buffer_pool(size=1, bound=False)
    drp = persona_ops.bwa_read_pool(size=0, bound=False)
    mmap_pool = persona_ops.m_map_pool(size=10, bound=False, name="file_mmap_buffer_pool")

    parsed_chunks = tf.contrib.persona.persona_in_pipe(metadata_path=metadata_json, dataset_dir=dataset_dir, columns=["base", "qual"], key=None, 
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

    record, key, num_records, first_ordinal = tf.train.batch_join_pdq(tensor_list_list=[e for e in aggregate_enqueue],
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

        regs_out = tf.train.batch_pdq(tensor_list=[aligned_regs, key, num_records, first_ordinal], batch_size=1, enqueue_many=False,
                                      num_dq_ops=1, name="align_to_pestat_queue")[0]

        # second stage infer insert size for chunk
        regs_out_us = regs_out[0]
        regs_stat = persona_ops.bwa_paired_end_stat(index_handle=bwa_index, options_handle=options, 
                                                    read=regs_out_us, name="bwa_pe_stat");

        to_enq = [regs_stat, regs_out[1], regs_out[2], regs_out[3]]
        regs_stat_out = tf.train.batch_pdq(tensor_list=to_enq, 
                                      batch_size=1, enqueue_many=False,
                                      num_dq_ops=1, name="pestat_to_finalizer_queue")[0]
       
        #final stage, pair selection
        regs_stat_us = regs_stat_out[0]
        aligned_result = persona_ops.bwa_finalize(index_handle=bwa_index, options_handle=options, num_threads=finalizer_threads,
                                      buffer_list_pool=blp, read=regs_stat_us, subchunk_size=subchunk_size, 
                                      max_read_size=400, name="bwa_finalize")
        
        aligned_results = tf.train.batch_pdq(tensor_list=[aligned_result, regs_stat_out[1], regs_stat_out[2], regs_stat_out[3]],
                                                batch_size=1, capacity=8, #this queue should always be empty!
                                                num_dq_ops=max(1, num_writers), name="results_to_sink")
    else:
        aligned = persona_ops.bwa_align_single(index_handle=bwa_index, options_handle=options, num_threads=aligner_threads,
                                      max_read_size=400, read=record, subchunk_size=subchunk_size, buffer_list_pool=blp, max_secondary=max_secondary, name="bwa_align")
        aligned_results = tf.train.batch_pdq(tensor_list=[aligned, key, num_records, first_ordinal], batch_size=1, enqueue_many=False,
                                      num_dq_ops=1, name="align_results_out")

    return aligned_results, bwa_index, options

def local_write_results(aligned_results, output_path, record_name, compress_output):
    ops = []
    for result_out, key_out, num_records, first_ordinal in aligned_results:
        results = tf.unstack()
        for i, res in enumerate(results):
          writer_op = persona_ops.parallel_column_writer(
              column_handle=res,
              record_type="results",
              record_id=record_name,
              extension="results" if i == 0 else "secondary{}".format(i),
              num_records=num_records,
              first_ordinal=first_ordinal,
              file_path=key_out, name="results_file_writer",
              compress=compress_output, output_dir=output_path
          )
          ops.append(writer_op)
    return ops

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
    import ipdb; ipdb.set_trace()

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
      
        #print(os.getpid())
        #import ipdb; ipdb.set_trace()

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
        metadata['reference'] = ref_list
        # adjust columns field with number of secondary results
        columns = ['base', 'qual', 'meta', 'results']
        for i in range(max_secondary):
          columns.append("secondary{}".format(i))
        metadata['columns'] = columns
        with open("test.json", 'w') as j:
          json.dump(metadata, j)



def run(args):
    all_ops = []
    indexes = []

    parallel_enqueue = args.enqueue; parallel_dequeue = args.parallel
    mmap_queue_length = args.mmap_queue; 
    local_path = args.local_path; paired = args.paired
    summary = args.summary


    aligned_results, bwa_index, options = create_ops(metadata_json=args.metadata_file, deep_verify=args.deep_verify,
                                                  num_writers=args.writers, subchunk_size=args.subchunking,
                                                  aligner_threads=args.aligner_threads, finalizer_threads=args.finalizer_threads, 
                                                  index_path=args.index_path, null_align=args.null,
                                                  paired=paired, dataset_dir=local_path, max_secondary=args.max_secondary)
    indexes.append(bwa_index)

    if args.writers == 0:
        results_out, key_out, _, _, = aligned_results[0]
        sink_op = persona_ops.buffer_list_sink(results_out)
        ops = sink_op
    else:
        print("adding writer")
        ops = tf.contrib.persona.persona_parallel_out_pipe(path=local_path, column=["results", "secondary0"], 
                                            write_list_list=aligned_results, record_id="persona_results", compress=args.compress) 

    all_ops = ops

    run_aligner(final_op=all_ops, indexes=indexes, summary=summary, null=True if args.null else False, metadata_path=args.metadata_file,
        max_secondary=args.max_secondary)

