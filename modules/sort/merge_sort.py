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
from tensorflow.contrib.persona import queues, pipeline

location_value = "location"

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

class VerifySortService(Service):
    def extract_run_args(self, args):
        return []
    def add_graph_args(self, parser):
        pass
    def add_run_args(self, parser):
        super().add_run_args(parser=parser)
        parser.add_argument("-d", "--dataset-dir", type=path_exists_checker(), required=True, help="Directory containing ALL of the chunk files")

    def get_shortname(self):
        return "verify"

    def output_dtypes(self, args):
        return (tf.dtypes.string,)

    def output_shapes(self, args):
        return (tf.tensor_shape.scalar(),)

    def distributed_capability(self):
        return False
    
    def make_graph(self, in_queue, args):
        records = args.dataset['records']
        first_record = records[0]
        chunk_size = first_record["last"] - first_record["first"]
        chunknames = []
        lens = []
        for record in records:
            chunknames.append(record['path'])
            lens.append(int(record['last']) - int(record['first']))

        print(lens)
        path = tf.constant(args.dataset_dir + '/')
        names = tf.constant(chunknames)
        sizes = tf.constant(lens)
        output = persona_ops.agd_verify_sort(path, names, sizes)

        return [], [output]

def get_inter_columns(order_by, columns):
    if order_by == location_value:
        inter_cols = ['results']
        for c in columns:
            if c != 'results':
                inter_cols.append(c)
    else:
        inter_cols = ['metadata']
        for c in columns:
            if c is not 'metadata':
                inter_cols.append(c)
    return inter_cols

def get_types_for_columns(columns):
    types = []
    typemap = {'results':'structured', 'secondary':'structured',
            'base':'base_compact', 'qual':'text', 'metadata':'text'}
    for c in columns:
        col = ''.join([i for i in c if not i.isdigit()])
        types.append(typemap[col])
    return types

def get_record_types_for_columns(order_by, columns):
    typemap = {'results':'structured', 'secondary':'structured',
            'base':'base_compact', 'qual':'text', 'metadata':'text'}
    if order_by == location_value:
        record_types = [{"type": "structured", "extension": "results"}]
        for c in columns:
            col = ''.join([i for i in c if not i.isdigit()])
            if c != 'results':
                record_types.append({"type": typemap[col], "extension":c})
    else:
        record_types = [{"type": "text", "extension": "metadata"}]
        for c in columns:
            col = ''.join([i for i in c if not i.isdigit()])
            if c != 'metadata':
                record_types.append({"type": typemap[col], "extension":c})
    return record_types



class SortCommonService(Service):
    columns = ["base", "qual", "metadata", "results"]
    inter_columns = ["results", "base", "qual", "metadata"]
    #merge_result_columns = ["results", "base", "qual", "metadata"]    just inter_columns
    #merge_meta_columns = ["metadata", "base", "qual", "results"]
    #records_type_location = ({"type": "structured", "extension": "results"},
                            #{"type": "base_compact", "extension": "base"},
                            #{"type": "text", "extension": "qual"},
                            #{"type": "text", "extension": "metadata"},
                            #)
    #records_type_metadata = ({"type": "text", "extension": "metadata"},
                            #{"type": "base_compact", "extension": "base"},
                            #{"type": "text", "extension": "qual"},
                            #{"type": "structured", "extension": "results"},
                            #)

    inter_file_name = "intermediate_file"

    def extract_run_args(self, args):
        dataset = args.dataset
        recs = [ a["path"] for a in dataset["records"] ]
        
        
        num_file_keys = len(recs)
        if num_file_keys < args.column_grouping:
            print("Column grouping factor too low! Setting to number of file keys ({})".format(num_file_keys))
            args.column_grouping = num_file_keys

        self.columns = dataset['columns']
        self.inter_columns = get_inter_columns(args.order_by, self.columns)
        print("sorting with columns {}".format(self.columns))
        print("inter columns is {}".format(self.inter_columns))
        return recs

    def add_graph_args(self, parser):
        # adds the common args to all graphs
        parser.add_argument("-r", "--sort-read-parallel", default=1, type=numeric_min_checker(minimum=1, message="read parallelism min for sort phase"),
                          help="total parallelism level for local read pipeline for sort phase")
        parser.add_argument("-p", "--sort-process-parallel", default=1, type=numeric_min_checker(minimum=1, message="process parallelism min for sort phase"), help="total parallelism level for local read pipeline for sort phase")
        parser.add_argument("-k", "--compress-parallel", default=1, type=numeric_min_checker(minimum=1, message="compress parallelism min for post merge write"), help="total parallelism level for compression")
        parser.add_argument("-c", "--column-grouping", default=5, help="grouping factor for parallel chunk sort",
                                  type=numeric_min_checker(minimum=1, message="column grouping min"))
        parser.add_argument("-s", "--sort-parallel", default=1, help="number of sorting pipelines to run in parallel",
                          type=numeric_min_checker(minimum=1, message="sorting pipeline min"))
        parser.add_argument("-w", "--write-parallel", default=1, help="number of writing pipelines to run in parallel",
                          type=numeric_min_checker(minimum=1, message="writing pipeline min"))
        parser.add_argument("--chunk", default=100000, type=numeric_min_checker(1, "need non-negative chunk size"), help="chunk size for final merge stage")
        parser.add_argument("-b", "--order-by", default="location", choices=["location", "metadata"], help="sort by this parameter [location | metadata]")

    def make_compressors(self, recs_and_handles, bufpool):

      #buf_pool = persona_ops.buffer_pool(size=0, bound=False, name="bufpool")

        for a in recs_and_handles:
            # out_tuple = [results, base, qual, meta, record_name, first_ord, num_recs, file_name]
            compressed_bufs = []
            for i, buf in enumerate(a[:len(self.inter_columns)]):
                compact = i == 1 # bases is always second column
                compressed_bufs.append(persona_ops.buffer_pair_compressor(buffer_pool=bufpool, buffer_pair=buf, pack=compact))

            compressed_matrix = tf.stack(compressed_bufs)
            yield [compressed_matrix] + a[len(self.inter_columns):]

    # chunks_to_merge is matrix of handles
    def make_merge_pipeline(self, args, record_name, chunks_to_merge, bpp):

        types = [dtypes.int32] + [dtypes.string]*len(self.inter_columns)
        shapes = [tensor_shape.scalar()] + [tensor_shape.vector(2)]*len(self.inter_columns)
        q = data_flow_ops.FIFOQueue(capacity=8, # big because who cares
                                    dtypes=types,
                                    shapes=shapes,
                                    name="merge_output_queue")

        #bpp = persona_ops.buffer_pair_pool(size=0, bound=False, name="local_read_merge_buffer_list_pool")

        if args.order_by == location_value:
            merge = persona_ops.agd_merge
        else:
            merge = persona_ops.agd_merge_metadata

        merge_op = merge(chunk_size=args.chunk,
                                 buffer_pair_pool=bpp,
                                 chunk_group_handles=chunks_to_merge,
                                 output_buffer_queue_handle=q.queue_ref,
                                 name="agd_local_merge")

        tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(q, [merge_op]))
   
        # num_recs, results, base, qual, meta
        #num_recs, results, base, qual, meta = q.dequeue()
        val = q.dequeue()
        num_recs = val[0]

        record_name_constant = constant_op.constant(record_name)
        first_ordinal = tf.Variable(-1 * args.chunk, dtype=dtypes.int64, name="first_ordinal")
        first_ord = first_ordinal.assign_add(math_ops.to_int64(args.chunk, name="first_ord_cast_to_64"), use_locking=True)
        first_ord_str = string_ops.as_string(first_ord, name="first_ord_string")
        file_name = string_ops.string_join([args.dataset_dir, "/", record_name_constant, first_ord_str], name="file_name_string_joiner")
       
        out_tuple = val[1:] + [record_name, first_ord, num_recs, file_name]

        return out_tuple
        
    def make_sort_pipeline(self, args, input_gen, buf_pool, bufpair_pool):

        ready_to_process = pipeline.join(upstream_tensors=input_gen,
                                         parallel=args.sort_process_parallel,
                                         capacity=4, # multiplied by some factor?
                                         multi=True, name="ready_to_process")
        # need to unpack better here
        multi_column_gen = list(pipeline.agd_reader_multi_column_pipeline(upstream_tensorz=ready_to_process, buffer_pool=buf_pool))
        # [ [base qual meta result], num_recs, first_ord, record_id ] 
        chunks_and_recs = []
        for chunks, num_recs, first_ord, record_id in multi_column_gen:
            entry = []
            for chunk in chunks:
                entry.append(chunk)
            entry.append(num_recs)
            chunks_and_recs.append(entry)

        ready = tf.train.batch_join(chunks_and_recs, batch_size=args.column_grouping, allow_smaller_final_batch=True, name="chunk_batcher")
       
        name_queue = pipeline.join([name_generator("intermediate_file")], parallel=args.sort_parallel, capacity=4, multi=False, name="inter_file_gen_q")
    
        #bpp = persona_ops.buffer_pair_pool(size=0, bound=False, name="local_read_buffer_pair_pool")

        if args.order_by == location_value:
            sorter = persona_ops.agd_sort
        else:
            sorter = persona_ops.agd_sort_metadata

        sorters = []
        for i in range(args.sort_parallel):
            #b, q, m, r, num = ready
            num = ready[-1]
            r = ready[0] # the sort predicate column must be first
            cols = tf.stack(ready[1:-1])
            superchunk_matrix, num_recs = sorter(buffer_pair_pool=bufpair_pool,
                              results_handles=r, column_handles=cols,
                              num_records=num, name="local_read_agd_sort")
            # super chunk is r, b, q, m
            sorters.append( [ superchunk_matrix, num_recs, name_queue[i] ] )

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
        parser.add_argument("-d", "--dataset-dir", type=path_exists_checker(), help="Directory containing ALL of the chunk files")
    
    def on_finish(self, args, results):
        # remove the intermediate files
        for f in os.listdir(args.dataset_dir):
            if self.inter_file_name in f:
                os.remove(os.path.join(args.dataset_dir, f))

        # add or change the sort order 
        #meta = "test.json"
        args.dataset['sort'] = 'coordinate' if args.order_by == location_value else 'queryname'
        #for rec in args.dataset['records']:
        #    rec['path'] = rec['path'].split('_')[0] + "_out_" + str(rec['first'])
        for metafile in os.listdir(args.dataset_dir):
            if metafile.endswith(".json"):
                with open(os.path.join(args.dataset_dir, metadata), 'w+') as f:
                    json.dump(args.dataset, f, indent=4)
                break
        #print("results were {}".format(results))

class LocalSortService(LocalCommonService):
    """ A service to use the SNAP aligner with a local dataset """

    def get_shortname(self):
        return "local"

    def output_dtypes(self, args):
        return (tf.dtypes.string,)

    def output_shapes(self, args):
        return (tf.tensor_shape.scalar(),)

    def distributed_capability(self):
        return False

    def make_inter_writers(self, batch, output_dir, write_parallelism):
        single = pipeline.join(batch, parallel=write_parallelism, capacity=4, multi=True, name="writer_queue")
        types = get_types_for_columns(self.inter_columns)
        print("inter col types {}".format(types))
        #types = [ "structured", "base_compact", "text", "text"]
      
        # no uncompressed buffer pair writer yet
        writers = []
        for buf, num_recs, record_id in single:
            w = [] 
            bufs = tf.unstack(buf)
            for i, b in enumerate(bufs):
                result_key = string_ops.string_join([output_dir, "/", record_id, ".",  self.inter_columns[i]], name="key_string")
                
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
      
        #print(compressed_buf)
        # upstream_tensors: a list of tensor tuples of type: buffer_list_handle, record_id, first_ordinal, num_records, file_path
        #types = self.records_type_location if args.order_by == location_value else self.records_type_metadata
        types = get_record_types_for_columns(args.order_by, self.inter_columns)
        print("final write types {}".format(types))
        writers = pipeline.local_write_pipeline(upstream_tensors=[compressed_buf], compressed=True, record_types=types, name="local_write_pipeline")
        
        return writers


    def make_graph(self, in_queue, args):

        # TODO remove the _out when we are satisfied it works correctly
        rec_name = args.dataset['records'][0]['path'][:-1]   # assuming path name is chunk_file_{ordinal}
        print("Sorting {} chunks".format(len(args.dataset['records'])))

        parallel_key_dequeue = tuple(in_queue.dequeue() for _ in range(args.sort_read_parallel))

        # read_files: [(file_path, (mmaped_file_handles, a gen)) x N]
        mmap_pool = persona_ops.m_map_pool(name="mmap_pool", size=10, bound=False)

        read_files = list(list(a) for a in pipeline.local_read_pipeline(upstream_tensors=parallel_key_dequeue, columns=self.inter_columns, mmap_pool=mmap_pool))
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
        #if args.order_by == location_value:
            #merge_cols = self.merge_result_columns
        #else:
            #merge_cols = self.merge_meta_columns

        merge_files = list(list(a) for a in pipeline.local_read_pipeline(upstream_tensors=[full_path_scalar], sync=False, columns=merge_cols, mmap_pool=mmap_pool))
        stacked_chunks = []
        for f in merge_files:
            stacked_chunks.append(tf.stack(f))

        # batch all the intermediate superchunks that are now mmap'd
        chunks_to_merge = tf.train.batch(stacked_chunks, batch_size=num_inter_files, name="mapped_inter_files_to_merge")
        merge_tuple = self.make_merge_pipeline(args=args, chunks_to_merge=chunks_to_merge, record_name=rec_name, bpp=bpp)
        # out_tuple = [results, base, qual, meta, record_name, first_ord, num_recs, file_name]
      
        compress_queue = pipeline.join(merge_tuple, capacity=4, parallel=args.compress_parallel, multi=False, name="to_compress")

        compressed_bufs = list(self.make_compressors(compress_queue, buf_pool))
        print(compressed_bufs)
        writers = list(list(a) for a in self.make_writers(args, compressed_bufs))

        return writers, []
        
class CephCommonService(SortCommonService):

    def input_dtypes(self, args):
        """ Ceph services require the key and the pool name """
        return [tf.string] * 2

    def input_shapes(self, args):
        """ Ceph services require the key and the pool name """
        return [tf.TensorShape([])] * 2

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

class CephSortService(CephCommonService):
    """ A service to use the snap aligner with a ceph dataset """

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
        :return: [ key, namespace, num_records, first_ordinal, record_id, full_path ]
        """
        if args.ceph_pool_name is None:
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
        writer_outputs = pipeline.ceph_aligner_write_pipeline(
            upstream_tensors=to_writer_gen,
            user_name=args.ceph_user_name,
            cluster_name=args.ceph_cluster_name,
            ceph_conf_path=args.ceph_conf_path,
            compressed=args.compress,
            pool_name=args.ceph_pool_name
        )
        output_tensors = (b+(a,) for a,b in zip(writer_outputs, ((key, namespace, num_records, first_ordinal, record_id)
                                                                 for _, num_records, first_ordinal, record_id, key, namespace in aligner_results)))
        return output_tensors, run_first
