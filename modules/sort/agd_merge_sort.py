from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as itt

from tensorflow.python.ops import gen_user_ops
from tensorflow.python.ops.gen_user_ops import *

from tensorflow.python.framework import ops, tensor_shape, common_shapes, constant_op, dtypes
from tensorflow.python.ops import io_ops, variables, string_ops, array_ops, data_flow_ops, math_ops, control_flow_ops
from tensorflow.python import training as train
from tensorflow.python.training import queue_runner

import os
import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()

location_value = "location"

"""
User ops specifically pertaining to agd merge sort

Contains convenience methods for creating common patterns of the user_ops
for agd_merge_sort operation.

You may connect these yourself based on tf.user_ops (the parent module)
"""

def name_generator(base_name, separator="-"):
    """
    Given a basename, defines an op that will generate intermediate unique names
    based on the base_name parameter.

    The suffix will be separated with `separator`, and will start counting from 0.
    """
    start_var = variables.Variable(-1)
    incr_var = start_var.assign_add(1)
    var_as_string = string_ops.as_string(incr_var)
    if not isinstance(base_name, ops.Tensor):
        base_name = constant_op.constant(str(base_name), dtype=dtypes.string,
                                         shape=tensor_shape.scalar(), name="name_generator_base")
    return string_ops.string_join([base_name, var_as_string], separator=separator, name="name_generator")

def _key_maker(file_keys, intermediate_file_prefix, column_grouping_factor, parallel_read):
    num_file_keys = len(file_keys)
    if num_file_keys < column_grouping_factor:
        print("Column grouping factor too low! Setting to number of file keys ({})".format(num_file_keys))
        column_grouping_factor = num_file_keys
    if num_file_keys % column_grouping_factor != 0:
        print("Column grouping factor not an even divisor of num_file_keys! ({})".format(num_file_keys))
        while num_file_keys % column_grouping_factor != 0:
            column_grouping_factor -= 1
        print("Reducing column grouping factor to {}".format(column_grouping_factor))
    extra_keys = (column_grouping_factor - (len(file_keys) % column_grouping_factor)) % column_grouping_factor
    print("extra keys: {}".format(extra_keys))
    if extra_keys > 0:
        file_keys = list(itt.chain(file_keys, itt.repeat("", extra_keys)))

    string_producer = train.input.string_input_producer(file_keys, num_epochs=1, shuffle=False)
    sp_output = string_producer.dequeue()

    batched_output = train.input.batch_pdq([sp_output], batch_size=column_grouping_factor, num_dq_ops=1)
    if column_grouping_factor == 1:
        batched_output = [tf.stack((bo,)) for bo in batched_output]
    intermediate_name = name_generator(base_name=intermediate_file_prefix)

    # TODO parallelism can be specified here
    paired_output = train.input.batch_pdq([batched_output[0], intermediate_name], batch_size=1, num_dq_ops=parallel_read, name="keys_and_intermediate")
    return paired_output

def _make_ceph_read_pipeline(key_batch, cluster_name, user_name, pool_name, ceph_conf_path, read_size, buffer_pool_handle):
    suffix_sep = tf.constant(".")
    base_suffix = tf.constant("base")
    qual_suffix = tf.constant("qual")
    meta_suffix = tf.constant("metadata")
    result_suffix = tf.constant("results")
    bases = []
    quals = []
    metas = []
    results = []
    split_batch = array_ops.unstack(key_batch)

    for k in split_batch:
        bases.append(string_ops.string_join([k, suffix_sep, base_suffix]))
        quals.append(string_ops.string_join([k, suffix_sep, qual_suffix]))
        metas.append(string_ops.string_join([k, suffix_sep, meta_suffix]))
        results.append(string_ops.string_join([k, suffix_sep, result_suffix]))

    base_bufs = []
    qual_bufs = []
    meta_bufs = []
    result_bufs = []

    # we take [0] of CephReader here because we don't need the filename it outputs
    for b in bases:
        bb = persona_ops.ceph_reader(cluster_name=cluster_name, user_name=user_name, pool_name=pool_name,
                                  ceph_conf_path=ceph_conf_path, read_size=read_size, buffer_handle=buffer_pool_handle,
                                  queue_key=b, name=None)[0]
        bbe = tf.expand_dims(bb, 0)
        base_bufs.append(bbe)
    for q in quals:
        qq = persona_ops.ceph_reader(cluster_name=cluster_name, user_name=user_name, pool_name=pool_name,
                                  ceph_conf_path=ceph_conf_path, read_size=read_size, buffer_handle=buffer_pool_handle,
                                  queue_key=q, name=None)[0]
        qqe = tf.expand_dims(qq, 0)
        qual_bufs.append(qqe)
    for m in metas:
        mm = persona_ops.ceph_reader(cluster_name=cluster_name, user_name=user_name, pool_name=pool_name,
                                  ceph_conf_path=ceph_conf_path, read_size=read_size, buffer_handle=buffer_pool_handle,
                                  queue_key=m, name=None)[0]
        mme = tf.expand_dims(mm, 0)
        meta_bufs.append(mme)
    for r in results:
        rr = persona_ops.ceph_reader(cluster_name=cluster_name, user_name=user_name, pool_name=pool_name,
                                  ceph_conf_path=ceph_conf_path, read_size=read_size, buffer_handle=buffer_pool_handle,
                                  queue_key=r, name=None)[0]
        rre = tf.expand_dims(rr, 0)
        result_bufs.append(rre)

    return tf.concat(base_bufs, 0), tf.concat(qual_bufs, 0), tf.concat(meta_bufs, 0), tf.concat(result_bufs, 0)


def _make_read_pipeline(key_batch, local_directory, mmap_pool_handle):
    suffix_sep = tf.constant(".")
    base_suffix = tf.constant("base")
    qual_suffix = tf.constant("qual")
    meta_suffix = tf.constant("metadata")
    result_suffix = tf.constant("results")
    bases = []
    quals = []
    metas = []
    results = []
    split_batch = array_ops.unstack(key_batch)

    for k in split_batch:
        bases.append(string_ops.string_join([k, suffix_sep, base_suffix]))
        quals.append(string_ops.string_join([k, suffix_sep, qual_suffix]))
        metas.append(string_ops.string_join([k, suffix_sep, meta_suffix]))
        results.append(string_ops.string_join([k, suffix_sep, result_suffix]))

    reads, names = persona_ops.file_m_map(filename=bases[0], pool_handle=mmap_pool_handle, local_prefix=local_directory, synchronous=False, name="base_mmap")

    for b in bases[1:]:
        reads, names = persona_ops.staged_file_map(filename=b, upstream_refs=reads, upstream_names=names, pool_handle=mmap_pool_handle,
                                                        local_prefix=local_directory, synchronous=False, name="base_staged_mmap")
    for q in quals:
        reads, names = persona_ops.staged_file_map(filename=q, upstream_refs=reads, upstream_names=names, pool_handle=mmap_pool_handle,
                                                        local_prefix=local_directory, synchronous=False, name="qual_staged_mmap")
    for m in metas:
        reads, names = persona_ops.staged_file_map(filename=m, upstream_refs=reads, upstream_names=names, pool_handle=mmap_pool_handle,
                                                        local_prefix=local_directory, synchronous=False, name="meta_staged_mmap")
    for r in results:
        reads, names = persona_ops.staged_file_map(filename=r, upstream_refs=reads, upstream_names=names, pool_handle=mmap_pool_handle,
                                                        local_prefix=local_directory, synchronous=False, name="result_staged_mmap")

    base_reads, qual_reads, meta_reads, result_reads = tf.split(reads, 4)
    return base_reads, qual_reads, meta_reads, result_reads

def _make_sorters(batch, buffer_list_pool, order_by):
    # FIXME this needs the number of records
    for b, q, m, r, num_records, im_name in batch:
        if order_by == location_value:
            yield persona_ops.agd_sort(buffer_list_pool=buffer_list_pool,
                              results_handles=r, bases_handles=b,
                              qualities_handles=q, metadata_handles=m,
                              num_records=num_records, name="local_read_agd_sort"), im_name
        else:
            yield persona_ops.agd_sort_metadata(buffer_list_pool=buffer_list_pool,
                              results_handles=r, bases_handles=b,
                              qualities_handles=q, metadata_handles=m,
                              num_records=num_records, name="local_read_agd_sort"), im_name

def _make_agd_batch(ready_batch, buffer_pool):
    for b, q, m, r, inter_name in ready_batch:
        base_reads, base_num_records, base_first_ordinals = persona_ops.agd_reader(verify=False,
                                                                          unpack=False,
                                                                          buffer_pool=buffer_pool,
                                                                          file_handle=b,
                                                                          name="base_reader")
        qual_reads, qual_num_records, qual_first_ordinals = persona_ops.agd_reader(verify=False,
                                                                          buffer_pool=buffer_pool,
                                                                          file_handle=q,
                                                                          name="qual_reader")
        meta_reads, meta_num_records, meta_first_ordinals = persona_ops.agd_reader(verify=False,
                                                                          buffer_pool=buffer_pool,
                                                                          file_handle=m,
                                                                          name="meta_reader")
        result_reads, result_num_records, result_first_ordinals = persona_ops.agd_reader(verify=False,
                                                                                buffer_pool=buffer_pool,
                                                                                file_handle=r,
                                                                                name="result_reader")
        # TODO we should have some sort of verification on ordinals here!
        yield base_reads, qual_reads, meta_reads, result_reads, base_num_records, inter_name

def _make_writers(results_batch, output_dir):
    first_ordinal = constant_op.constant(0, dtype=dtypes.int64) # first ordinal doesn't matter for the sort phase
    for column_handle, num_records, im_name in results_batch:
        writer, first_o_passthru = persona_ops.agd_write_columns(record_id="fixme",
                                                       record_type=["base", "qual", "metadata", "results"],
                                                       column_handle=column_handle,
                                                       output_dir=output_dir + "/",
                                                       file_path=im_name,
                                                       first_ordinal=first_ordinal,
                                                       num_records=num_records,
                                                       name="agd_column_writer")
        yield writer # writes out the file path key (full path)

def _make_ceph_writers(results_batch, cluster_name, user_name, pool_name, ceph_conf_path):
    first_ordinal = constant_op.constant(0, dtype=dtypes.int64) # first ordinal doesn't matter for the sort phase
    for column_handle, num_records, im_name in results_batch:
        writer, first_o_passthru = persona_ops.agd_ceph_write_columns(cluster_name=cluster_name,
                                                       user_name=user_name,
                                                       pool_name=pool_name,
                                                       ceph_conf_path=ceph_conf_path,
                                                       record_id="fixme",
                                                       record_type=["base", "qual", "metadata", "results"],
                                                       column_handle=column_handle,
                                                       file_path=im_name,
                                                       first_ordinal=first_ordinal,
                                                       num_records=num_records,
                                                       name="agd_column_writer")
        yield writer, num_records # writes out the file path key (full path)

# TODO I'm not sure what to do about the last param
def local_sort_pipeline(file_keys, local_directory, outdir=None, intermediate_file_prefix="intermediate_file",
                        column_grouping_factor=5, parallel_read=1, parallel_process=1, parallel_sort=1, order_by=location_value):
    """
    file_keys: a list of Python strings of the file keys, which you should extract from the metadata file
    local_directory: the "base path" from which these should be read
    column_grouping_factor: the number of keys to put together
    parallel_process: the parallelism for processing records (reading, decomp)
    parallel_read: the number of parallel read pipelines
    parallel_sort: the number of parallel sort operations
    """
    if parallel_read < 1:
        raise Exception("parallel_read must be >1. Got {}".format(parallel_read))
    key_producers = _key_maker(file_keys=file_keys, intermediate_file_prefix=intermediate_file_prefix,
                               parallel_read=parallel_read, column_grouping_factor=column_grouping_factor)
    mapped_file_pool = persona_ops.m_map_pool(size=0, bound=False, name="local_read_mmap_pool")
    read_pipelines = [(_make_read_pipeline(key_batch=kp[0], local_directory=local_directory, mmap_pool_handle=mapped_file_pool),
                       kp[1]) for kp in key_producers]

    ready_record_batch = train.input.batch_join_pdq([tuple(k[0])+(k[1],) for k in read_pipelines], num_dq_ops=parallel_process,
                                                    batch_size=1, capacity=8, name="ready_record_queue")

    # now the AGD parallel stage
    bp = persona_ops.buffer_pool(size=0, bound=False, name="local_read_buffer_pool")
    processed_record_batch = _make_agd_batch(ready_batch=ready_record_batch, buffer_pool=bp)

    batched_processed_records = train.input.batch_join_pdq([a for a in processed_record_batch],
                                                           batch_size=1, num_dq_ops=parallel_sort, capacity=8,
                                                           name="sortable_ready_queue")

    blp = persona_ops.buffer_list_pool(size=0, bound=False, name="local_read_buffer_list_pool")

    sorters = _make_sorters(batch=batched_processed_records, buffer_list_pool=blp, order_by=order_by)

    batched_results = train.input.batch_join_pdq([a[0] + (a[1],) for a in sorters], num_dq_ops=1,
                                                 batch_size=1, name="sorted_im_files_queue")

    if outdir is None:
        outdir = local_directory
    intermediate_keys_records = _make_writers(results_batch=batched_results, output_dir=outdir)

    recs = [rec for rec in intermediate_keys_records]
    all_im_keys = train.input.batch_pdq(recs, num_dq_ops=1,
                                        batch_size=1, name="intermediate_key_queue")

    return all_im_keys

def ceph_sort_pipeline(file_keys, cluster_name, user_name, pool_name, ceph_conf_path, ceph_read_size, order_by, intermediate_file_prefix="intermediate_file",
                        column_grouping_factor=5, parallel_read=1, parallel_process=1, parallel_sort=1, parallel_write=1):
    """
    file_keys: a list of Python strings of the file keys, which you should extract from the metadata file
    cluster_name, user_name, pool_name, ceph_conf_path: Ceph parameters
    column_grouping_factor: the number of keys to put together
    parallel_process: the parallelism for processing records (reading, decomp)
    parallel_read: the number of parallel read pipelines
    parallel_sort: the number of parallel sort operations
    """
    if parallel_read < 1:
        raise Exception("parallel_read must be >1. Got {}".format(parallel_read))
    key_producers = _key_maker(file_keys=file_keys, intermediate_file_prefix=intermediate_file_prefix,
                               parallel_read=parallel_read, column_grouping_factor=column_grouping_factor)

    bp = persona_ops.buffer_pool(size=0, bound=False, name="ceph_read_buffer_pool")
    read_pipelines = [(_make_ceph_read_pipeline(key_batch=kp[0], cluster_name=cluster_name, user_name=user_name,
                       pool_name=pool_name, ceph_conf_path=ceph_conf_path, read_size=ceph_read_size, buffer_pool_handle=bp),
                       kp[1]) for kp in key_producers]

    ready_record_batch = train.input.batch_join_pdq([tuple(k[0])+(k[1],) for k in read_pipelines], num_dq_ops=parallel_process,
                                                    batch_size=1, capacity=8, name="ready_record_queue")
    # now the AGD parallel stage
    processed_record_batch = _make_agd_batch(ready_batch=ready_record_batch, buffer_pool=bp)

    batched_processed_records = train.input.batch_join_pdq([a for a in processed_record_batch],
                                                           batch_size=1, num_dq_ops=parallel_sort, capacity=8,
                                                           name="sortable_ready_queue")

    blp = persona_ops.buffer_list_pool(size=0, bound=False, name="ceph_read_buffer_list_pool")

    sorters = _make_sorters(batch=batched_processed_records, buffer_list_pool=blp, order_by=order_by)

    batched_results = train.input.batch_join_pdq([a[0] + (a[1],) for a in sorters], num_dq_ops=parallel_write,
                                                 batch_size=1, name="sorted_im_files_queue")

    intermediate_keys_records = _make_ceph_writers(results_batch=batched_results,
                                                   cluster_name=cluster_name,
                                                   user_name=user_name,
                                                   pool_name=pool_name,
                                                   ceph_conf_path=ceph_conf_path)

    recs = [rec for rec in intermediate_keys_records]
    all_im_keys = train.input.batch_join_pdq(recs, num_dq_ops=1,
                                             batch_size=1, name="intermediate_key_queue")

    return all_im_keys

### All the methods for creating the local merge pipeline

def _make_merge_read_records(key_outs, in_dir, mmap_pool_handle, order_by):
    suffix_sep = tf.constant(".")
    base_suffix = tf.constant("base")
    qual_suffix = tf.constant("qual")
    meta_suffix = tf.constant("metadata")
    result_suffix = tf.constant("results")
    # dictated by the merge op
    if order_by == location_value:
        suffix_order = [result_suffix, base_suffix, qual_suffix, meta_suffix]
    else:
        suffix_order = [meta_suffix, base_suffix, qual_suffix, result_suffix]

    def make_single_chunk_read(im_name):
        appended_names = [string_ops.string_join([im_name, suffix_sep, a]) for a in suffix_order]
        reads, names = persona_ops.file_m_map(filename=appended_names[0], local_prefix=in_dir,
                                    pool_handle=mmap_pool_handle, name="result_mmap")
        accum = [(reads, names)]
        for column in appended_names[1:]:
            prior_reads, prior_names = accum[-1]
            reads, names = persona_ops.staged_file_map(filename=column,
                                             upstream_refs=prior_reads,
                                             upstream_names=prior_names,
                                             pool_handle=mmap_pool_handle,
                                             local_prefix=in_dir,
                                             name="merge_column_mmap")
            accum.append((reads, names))
        return accum[-1][0]

    for key_out in key_outs:
        split_records = array_ops.unstack(key_out)
        yield [make_single_chunk_read(im_name=im_name) for im_name in split_records]

def _make_processed_records(ready_read_records, buffer_pool):
    def process_ready_row(interm_columns):
        columns_split = tf.unstack(interm_columns)
        return zip(*(persona_ops.agd_reader(verify=False, unpack=False,
                                   pool_handle=buffer_pool,
                                   file_handle=column,
                                   name="column_agd_reader") for column in columns_split))

    for interm_columns in ready_read_records:
        readss, num_recordss, first_ordinalss = process_ready_row(interm_columns=interm_columns)
        yield [nr[0] for nr in num_recordss], tf.stack(readss)

def local_merge_pipeline(intermediate_keys, in_dir, record_name, write_parallel, outdir=None, chunk_size=100000, order_by=location_value):
    if chunk_size < 1:
        raise Exception("Need strictly non-negative chunk size. Got {}".format(chunk_size))
    if write_parallel < 1:
        raise Exception("Need strictly >1 write parallelism. Got {}".format(write_parallel))
    key_producer = train.input.input_producer([intermediate_keys],
                                              # this element_shape specification isn't necessary, but it's a good double-check
                                              #element_shape=tensor_shape.vector(len(intermediate_keys)),
                                              capacity=1,
                                              shuffle=False,
                                              num_epochs=1,
                                              name="merge_key_producer")
    key_output = key_producer.dequeue()
    key_outs = train.input.batch_pdq([key_output], batch_size=1, num_dq_ops=1)
    mapped_file_pool = persona_ops.m_map_pool(size=0, bound=False, name="local_read_mmap_pool")
    ready_read_records = _make_merge_read_records(key_outs=key_outs, in_dir=in_dir,
                                                  mmap_pool_handle=mapped_file_pool, order_by=order_by)

    # Note: we are skippng the processing part here because we just use the big buffers directly

    merge_ready_queue = train.input.batch_join_pdq([(tf.stack(p),) for p in ready_read_records],
                                                   num_dq_ops=1, batch_size=1, name="merge_ready_queue")
    q = data_flow_ops.FIFOQueue(capacity=100000, # big because who cares
                                dtypes=[dtypes.int32, dtypes.string],
                                shapes=[tensor_shape.scalar(), tensor_shape.vector(2)],
                                name="merge_output_queue")

    blp = persona_ops.buffer_list_pool(size=0, bound=False, name="local_read_merge_buffer_list_pool")
    chunk_group_handles = merge_ready_queue[0]

    if order_by == location_value:
        merge_op = persona_ops.agd_merge(chunk_size=chunk_size,
                                 buffer_list_pool=blp,
                                 chunk_group_handles=chunk_group_handles,
                                 output_buffer_queue_handle=q.queue_ref,
                                 name="agd_local_merge")
    else:
        merge_op = persona_ops.agd_merge_metadata(chunk_size=chunk_size,
                                          buffer_list_pool=blp,
                                          chunk_group_handles=chunk_group_handles,
                                          output_buffer_queue_handle=q.queue_ref,
                                          name="agd_local_merge_metadata")

    queue_runner.add_queue_runner(queue_runner.QueueRunner(q, [merge_op]))

    if outdir is None:
        outdir = in_dir

    # FIXME if you don't have at least chunk_size records in your dataset, this will cause underflow
    # this is a hack!
    first_ordinal = tf.Variable(-1 * chunk_size, dtype=dtypes.int64, name="first_ordinal")

    record_name_constant = constant_op.constant(record_name+"-")
    num_recs, buffer_list_handle = q.dequeue()
    first_ord = first_ordinal.assign_add(math_ops.to_int64(num_recs, name="first_ord_cast_to_64"), use_locking=True)
    first_ord_str = string_ops.as_string(first_ord, name="first_ord_string")
    file_name = string_ops.string_join([record_name_constant, first_ord_str], name="file_name_string_joiner")
    write_join_tensor = control_flow_ops.tuple(tensors=[buffer_list_handle, num_recs, first_ord, file_name], name="write_join_tensor")
    write_join_queue = train.input.batch_pdq(write_join_tensor, num_dq_ops=write_parallel, batch_size=1, name="write_join_queue", capacity=write_parallel)

    if order_by == location_value:
        record_type = ["results", "base", "qual", "metadata"]
    else:
        record_type = ["metadata", "base", "qual", "results"]

    final_write_out = []
    for buff_list, n_recs, first_o, file_key in write_join_queue:
        file_key_passthru, first_o_passthru = persona_ops.agd_write_columns(record_id=record_name,
                                                                    record_type=record_type,
                                                                    column_handle=buff_list,
                                                                    output_dir=outdir+"/",
                                                                    file_path=file_key,
                                                                    first_ordinal=first_o,
                                                                    num_records=n_recs,
                                                                    name="agd_column_writer_merge")
        final_write_out.append([file_key_passthru, first_o_passthru, n_recs])
    #return final_write_out[0]

    sink_queue = train.input.batch_join_pdq(final_write_out, capacity=1, num_dq_ops=1, batch_size=1, name="final_sink_queue")
    return sink_queue[0]

def ceph_merge_pipeline(intermediate_keys, record_name, num_records, cluster_name, user_name, pool_name, output_pool_name,
        ceph_conf_path, chunk_size=100000, parallel_write=1):
    if chunk_size < 1:
        raise Exception("Need strictly non-negative chunk size. Got {}".format(chunk_size))

    blp = persona_ops.buffer_list_pool(size=0, bound=False, name="ceph_read_merge_buffer_list_pool")

    new_chunk_handle, num_recs = persona_ops.agd_ceph_merge(chunk_size=chunk_size,
                                                    intermediate_files=intermediate_keys,
                                                    num_records=num_records,
                                                    cluster_name=cluster_name,
                                                    user_name=user_name,
                                                    pool_name=pool_name,
                                                    ceph_conf_path=ceph_conf_path,
                                                    file_buf_size=10,
                                                    buffer_list_pool=blp)


    chunk_handle = train.input.batch_pdq([new_chunk_handle, num_recs], num_dq_ops=1,
                                         batch_size=1, capacity=8, name="chunk_handle_out_queue")


    # FIXME if you don't have at least chunk_size records in your dataset, this will cause underflow
    # this is a hack!
    first_ordinal = tf.Variable(-1 * chunk_size, dtype=dtypes.int64, name="first_ordinal")

    record_name_constant = constant_op.constant(record_name+"-")
    num_recs, buffer_list_handle = chunk_handle[0][1], chunk_handle[0][0]
    first_ord = first_ordinal.assign_add(math_ops.to_int64(num_recs, name="first_ord_cast_to_64"), use_locking=True)
    first_ord_str = string_ops.as_string(first_ord, name="first_ord_string")
    file_name = string_ops.string_join([record_name_constant, first_ord_str], name="file_name_string_joiner")
    write_join_tensor = control_flow_ops.tuple(tensors=[buffer_list_handle, num_recs, first_ord, file_name], name="write_join_tensor")
    write_join_queue = train.input.batch_pdq(write_join_tensor, num_dq_ops=parallel_write, batch_size=1, name="write_join_queue", capacity=1)

    final_write_out = []
    for buff_list, n_recs, first_o, file_key in write_join_queue:
        file_key_passthru, first_o_passthru = persona_ops.agd_ceph_write_columns(cluster_name=cluster_name,
                                                                         user_name=user_name,
                                                                         pool_name=output_pool_name,
                                                                         ceph_conf_path=ceph_conf_path,
                                                                         record_id=record_name,
                                                                         record_type=["results", "base", "qual", "metadata"],
                                                                         column_handle=buff_list,
                                                                         file_path=file_key,
                                                                         first_ordinal=first_o,
                                                                         num_records=n_recs,
                                                                         name="agd_column_writer_merge")
        final_write_out.append([file_key_passthru, first_o_passthru, n_recs])
    #return final_write_out[0]

    sink_queue = train.input.batch_join_pdq(final_write_out, capacity=1, num_dq_ops=1, batch_size=1, name="final_sink_queue")
    return sink_queue[0]
