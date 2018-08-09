# Modified version of modules/sort/merge_sort.py to sort proteins based on their sequence length
# Milad, Aug. 2018

from tensorflow.contrib.persona import queues, pipeline

import os
import json
import math
from ..common.service import Service
from common.parse import numeric_min_checker, path_exists_checker, non_empty_string_checker
from tensorflow.python.ops import io_ops, variables, string_ops, array_ops, data_flow_ops, math_ops, control_flow_ops
from tensorflow.python.framework import ops, tensor_shape, common_shapes, constant_op, dtypes
from ..common import parse

import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()

sorting_criterion = 'prot'  # equivalent of location_value in (DNA read) sort
typemap = {'prot': 'text', 'metadata': 'text'}  # protein sequence column must be named 'prot'


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
        base_name = tf.constant(str(base_name), dtype=dtypes.string,
                                shape=tensor_shape.scalar(), name="name_generator_base")
    return tf.string_join([base_name, var_as_string], separator=separator, name="name_generator")


def get_inter_columns(order_by, columns):  # presuming order_by is a valid column name!
    return [order_by] + list(filter(lambda c: c != order_by, columns))


def get_types_for_columns(columns):
    types = []
    # typemap = {'results': 'structured', 'secondary': 'structured',
    #            'base': 'base_compact', 'qual': 'text', 'metadata': 'text'}
    for c in columns:
        col = ''.join([i for i in c if not i.isdigit()])
        types.append(typemap[col])
    return types


def get_record_types_for_columns(order_by, columns):
    record_types = [{"type": "text", "extension": "prot"}]
    for c in columns:
        col = ''.join([i for i in c if not i.isdigit()])  # why though?
        if c != order_by:  # milad: instead of 'metadata'
            record_types.append({"type": typemap[col], "extension": c})
    return record_types


class SortCommonService(Service):
    columns = ["prot", "metadata"]

    inter_file_name = "intermediate_file"

    def distributed_capability(self):
        return False

    def extract_run_args(self, args):  # TODO check if args.order_by is set
        dataset = args.dataset
        recs = [a["path"] for a in dataset["records"]]

        rec = dataset["records"][0]
        args.chunk = int(rec["last"]) - int(rec["first"])
        # print("Chunk size is {}".format(args.chunk))
        num_file_keys = len(recs)
        if num_file_keys < args.column_grouping:
            # print("Column grouping factor too low! Setting to number of file keys ({})".format(num_file_keys))
            args.column_grouping = num_file_keys

        self.columns = dataset['columns']
        self.inter_columns = get_inter_columns(args.order_by, self.columns)
        # print("sorting with columns {}".format(self.columns))
        # print("inter columns is {}".format(self.inter_columns))
        return recs

    def add_graph_args(self, parser):  # TODO proper description for the arguments
        # adds the common args to all graphs
        parser.add_argument("-r", "--sort-read-parallel", default=1,
                            type=numeric_min_checker(minimum=1, message="read parallelism min for sort phase"),
                            help="total parallelism level for local read pipeline for sort phase")

        parser.add_argument("-p", "--sort-process-parallel", default=1,
                            type=numeric_min_checker(minimum=1, message="sort process parallel message!"),
                            help="sort process parallel help")

        # parser.add_argument("-p", "--sort-process-parallel", default=1,
        #                     type=numeric_min_checker(minimum=1, message="process parallelism min for sort phase"),
        #                     help="total parallelism level for local read pipeline for sort phase")
        parser.add_argument("-k", "--compress-parallel", default=1, type=numeric_min_checker(minimum=1,
                                                                                             message="compress parallelism min for post merge write"),
                            help="total parallelism level for compression")
        parser.add_argument("-c", "--column-grouping", default=5, help="grouping factor for parallel chunk sort",
                            type=numeric_min_checker(minimum=1, message="column grouping min"))
        parser.add_argument("-s", "--sort-parallel", default=1, help="number of sorting pipelines to run in parallel",
                            type=numeric_min_checker(minimum=1, message="sorting pipeline min"))
        parser.add_argument("-w", "--write-parallel", default=1, help="number of writing pipelines to run in parallel",
                            type=numeric_min_checker(minimum=1, message="writing pipeline min"))
        parser.add_argument("--chunk", default=100000, type=numeric_min_checker(1, "need non-negative chunk size"), help="chunk size for final merge stage")
        parser.add_argument("-b", "--order-by", default="prot", choices=["prot", "metadata"],
                            help="sort by this parameter [prot | metadata]")

    def make_compressors(self, recs_and_handles, bufpool):
        # buf_pool = persona_ops.buffer_pool(size=0, bound=False, name="bufpool")
        for a in recs_and_handles:
            # out_tuple = [results, base, qual, meta, record_name, first_ord, num_recs, file_name]
            compressed_bufs = []
            for i, buf in enumerate(a[:len(self.inter_columns)]):
                # compact = i == 1  # bases is always second column
                # compact = i == 0  # milad: pack = False always for protein
                compressed_bufs.append(
                    persona_ops.buffer_pair_compressor(buffer_pool=bufpool, buffer_pair=buf, pack=False))
                    # persona_ops.buffer_pair_compressor(buffer_pool=bufpool, buffer_pair=buf, pack=compact))
                # pack is hardcoded to be false

            compressed_matrix = tf.stack(compressed_bufs)
            yield [compressed_matrix] + a[len(self.inter_columns):]

    # chunks_to_merge is matrix of handles
    def make_merge_pipeline(self, args, record_name, chunks_to_merge, bpp):
        types = [dtypes.int32] + [dtypes.string] * len(self.inter_columns)
        shapes = [tensor_shape.scalar()] + [tensor_shape.vector(2)] * len(self.inter_columns)
        q = data_flow_ops.FIFOQueue(capacity=8,  # big because who cares
                                    dtypes=types,
                                    shapes=shapes,
                                    name="merge_output_queue")

        # bpp = persona_ops.buffer_pair_pool(size=0, bound=False, name="local_read_merge_buffer_list_pool")

        merge = persona_ops.agd_protein_merge

        merge_op = merge(chunk_size=args.chunk,
                         buffer_pair_pool=bpp,
                         chunk_group_handles=chunks_to_merge,
                         output_buffer_queue_handle=q.queue_ref,
                         name="agd_protein_merge")  # name="agd_local_merge"
        tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(q, [merge_op]))
        # num_recs, results, base, qual, meta
        # num_recs, results, base, qual, meta = q.dequeue()
        val = q.dequeue()
        num_recs = val[0]

        record_name_constant = constant_op.constant(record_name)
        first_ordinal = tf.Variable(-1 * args.chunk, dtype=dtypes.int64, name="first_ordinal")
        first_ord = first_ordinal.assign_add(math_ops.to_int64(args.chunk, name="first_ord_cast_to_64"),
                                             use_locking=True)
        first_ord_str = string_ops.as_string(first_ord, name="first_ord_string")

        #file_name = string_ops.string_join(["/scratch/mrazavi/proteins/output", "/", record_name_constant, first_ord_str],
        #                                   name="file_name_string_joiner")  # TODO remove this and use the one below
        file_name = string_ops.string_join([args.dataset_dir, "/", record_name_constant, first_ord_str],
                                           name="file_name_string_joiner")  # TODO cm for dbg

        out_tuple = val[1:] + [record_name, first_ord, num_recs, file_name]

        return out_tuple

    def make_sort_pipeline(self, args, input_gen, buf_pool, bufpair_pool):
        # print(args)
        ready_to_process = pipeline.join(upstream_tensors=input_gen,
                                         parallel=args.sort_process_parallel,
                                         capacity=4,  # multiplied by some factor?
                                         multi=True, name="ready_to_process")
        # need to unpack better here
        multi_column_gen = list(
            pipeline.agd_reader_multi_column_pipeline(upstream_tensorz=ready_to_process, buffer_pool=buf_pool))
        # [ [base qual meta result], num_recs, first_ord, record_id ]
        chunks_and_recs = []
        for chunks, num_recs, first_ord, record_id in multi_column_gen:
            entry = []
            for chunk in chunks:
                entry.append(chunk)
            entry.append(num_recs)
            chunks_and_recs.append(entry)

        ready = tf.train.batch_join(chunks_and_recs, batch_size=args.column_grouping, allow_smaller_final_batch=True,
                                    name="chunk_batcher")

        name_queue = pipeline.join([name_generator("intermediate_file")], parallel=args.sort_parallel, capacity=4,
                                   multi=False, name="inter_file_gen_q")

        # bpp = persona_ops.buffer_pair_pool(size=0, bound=False, name="local_read_buffer_pair_pool")

        sorter = persona_ops.agd_protein_sort

        sorters = []
        for i in range(args.sort_parallel):
            # b, q, m, r, num = ready
            num = ready[-1]
            r = ready[0]  # the sort predicate column must be first
            cols = tf.stack(ready[1:-1])
            superchunk_matrix, num_recs = sorter(buffer_pair_pool=bufpair_pool,
                                                 results_handles=r, column_handles=cols,
                                                 num_records=num, name="local_protein_sequence_agd_sort")
            # super chunk is r, b, q, m
            sorters.append([superchunk_matrix, num_recs, name_queue[i]])
        return sorters


class LocalCommonService(SortCommonService):
    def extract_run_args(self, args):
        dataset_dir = args.dataset_dir
        if dataset_dir is None:
            file_path = args.dataset[parse.filepath_key]
            dataset_dir = os.path.dirname(file_path)
        args.dataset_dir = dataset_dir

        return (os.path.join(dataset_dir, a) for a in super().extract_run_args(args=args))

    def add_run_args(self, parser):
        super().add_run_args(parser=parser)
        parser.add_argument("-d", "--dataset-dir", type=path_exists_checker(),
                            help="Directory containing ALL of the chunk files")

    def on_finish(self, args, results):
        # remove the intermediate files
        for f in os.listdir(args.dataset_dir):
            if self.inter_file_name in f:
                os.remove(os.path.join(args.dataset_dir, f))
        # add or change the sort order
        # meta = "test.json"

        # TODO milad check if this is correct
        args.dataset['sort'] = 'coordinate' if args.order_by == sorting_criterion else 'queryname'
        # args.dataset['sort'] = 'coordinate' if args.order_by == location_value else 'queryname'


        # for rec in args.dataset['records']:
        #    rec['path'] = rec['path'].split('_')[0] + "_out_" + str(rec['first'])
        for metafile in os.listdir(args.dataset_dir):
            if metafile.endswith(".json"):
                with open(os.path.join(args.dataset_dir, metafile), 'w+') as f:
                    json.dump(args.dataset, f, indent=4)
                break
        # print("results were {}".format(results))


class LocalSortService(LocalCommonService):
    """ A service to use the SNAP aligner with a local dataset """

    def get_shortname(self):
        return "prot_sort"

    def output_dtypes(self, args):
        return (tf.dtypes.string,)

    def output_shapes(self, args):
        return (tf.tensor_shape.scalar(),)

    def distributed_capability(self):
        return False

    def make_inter_writers(self, batch, output_dir, write_parallelism):
        single = pipeline.join(batch, parallel=write_parallelism, capacity=4, multi=True, name="writer_queue")
        types = get_types_for_columns(self.inter_columns)
        # print("inter col types {}".format(types))
        # types = [ "structured", "base_compact", "text", "text"]

        # no uncompressed buffer pair writer yet
        writers = []
        for buf, num_recs, record_id in single:
            w = []
            bufs = tf.unstack(buf)
            for i, b in enumerate(bufs):
                result_key = string_ops.string_join([output_dir, "/", record_id, ".", self.inter_columns[i]],
                                                    name="key_string")
                # print(self.inter_columns)
                result = persona_ops.agd_file_system_buffer_pair_writer(record_id=record_id,
                                                                        record_type=types[i],
                                                                        resource_handle=b,
                                                                        path=result_key,
                                                                        first_ordinal=0,
                                                                        num_records=tf.to_int32(num_recs))
                w.append(result)
            w.append(record_id)
            writers.append(w)
        return writers

    def make_writers(self, args, compressed_bufs):
        compressed_buf = pipeline.join(compressed_bufs, capacity=4, multi=True, parallel=1, name="final_write_queue")[0]

        # add parallelism here if necessary to saturate write bandwidth
        # [compressed_matrix, record_name, first_ord, num_recs, file_name]

        # print(compressed_buf)
        # upstream_tensors: a list of tensor tuples of type: buffer_list_handle, record_id, first_ordinal, num_records, file_path
        # types = self.records_type_location if args.order_by == location_value else self.records_type_metadata
        types = get_record_types_for_columns(args.order_by, self.inter_columns)
        # print("final write types {}".format(types))
        writers = pipeline.local_write_pipeline(upstream_tensors=[compressed_buf], compressed=True, record_types=types,
                                                name="local_write_pipeline")

        return writers

    def make_graph(self, in_queue, args):

        # TODO remove the _out when we are satisfied it works correctly
        rec_name = args.dataset['records'][0]['path'][:-1]  # assuming path name is chunk_file_{ordinal}
        # print("Sorting {} chunks".format(len(args.dataset['records'])))

        parallel_key_dequeue = tuple(in_queue.dequeue() for _ in range(args.sort_read_parallel))

        # read_files: [(file_path, (mmaped_file_handles, a gen)) x N]
        mmap_pool = persona_ops.m_map_pool(name="mmap_pool", size=10, bound=False)
        read_files = list(list(a) for a in pipeline.local_read_pipeline(upstream_tensors=parallel_key_dequeue,
                                                                        columns=self.inter_columns,
                                                                        mmap_pool=mmap_pool))
        # need to use tf.tuple to make sure that these are both made ready at the same time

        buf_pool = persona_ops.buffer_pool(size=0, bound=False, name="bufpool")
        bpp = persona_ops.buffer_pair_pool(size=0, bound=False, name="local_read_merge_buffer_list_pool")

        sorters = self.make_sort_pipeline(args=args, input_gen=read_files, buf_pool=buf_pool, bufpair_pool=bpp)

        writers = self.make_inter_writers(sorters, args.dataset_dir, args.write_parallel)

        inter_file_paths = pipeline.join(writers, parallel=1, capacity=3, multi=True, name="writer_queue")[0]
        inter_file_name = inter_file_paths[-1]

        num_inter_files = int(math.ceil(len(args.dataset['records']) / args.column_grouping))

        # these two queues form a barrier, to force downstream to wait until all intermediate superchunks are ready for merge
        # wait for num_inter_files
        f = tf.train.batch([inter_file_name], batch_size=num_inter_files, name="inter_file_batcher")
        # now output them one by one
        files = tf.train.batch([f], enqueue_many=True, batch_size=1, name="inter_file_output")
        full_path = tf.string_join([args.dataset_dir, "/", files])
        # needs to be scalar not shape [1] which seems pretty stupid ...
        full_path_scalar = tf.reshape(full_path, [])

        # may need to add disk read parallelism here
        merge_cols = self.inter_columns
        # if args.order_by == location_value:
        # merge_cols = self.merge_result_columns
        # else:
        # merge_cols = self.merge_meta_columns

        merge_files = list(list(a) for a in
                           pipeline.local_read_pipeline(upstream_tensors=[full_path_scalar], sync=False,
                                                        columns=merge_cols, mmap_pool=mmap_pool))
        stacked_chunks = []
        for f in merge_files:
            stacked_chunks.append(tf.stack(f))

        # batch all the intermediate superchunks that are now mmap'd
        chunks_to_merge = tf.train.batch(stacked_chunks, batch_size=num_inter_files, name="mapped_inter_files_to_merge")
        merge_tuple = self.make_merge_pipeline(args=args, chunks_to_merge=chunks_to_merge, record_name=rec_name,
                                               bpp=bpp)
        # out_tuple = [results, base, qual, meta, record_name, first_ord, num_recs, file_name]

        compress_queue = pipeline.join(merge_tuple, capacity=4, parallel=args.compress_parallel, multi=False,
                                       name="to_compress")

        compressed_bufs = list(self.make_compressors(compress_queue, buf_pool))
        # print(compressed_bufs)
        writers = list(list(a) for a in self.make_writers(args, compressed_bufs))
        # input("press enter")
        return writers, []



