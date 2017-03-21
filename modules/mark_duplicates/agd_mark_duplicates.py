from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from ..common.service import Service

persona_ops = tf.contrib.persona.persona_ops()

"""
ops specifically pertaining to agd mark duplicates

Contains convenience methods for creating common patterns for 
marking duplicates.

You may connect these yourself based on persona_ops (the parent module)
"""

class MarkDuplicateService(Service):
    """ A class representing a service module in Persona """
   
    #default inputs

    def output_dtypes(self):
        return []
    def output_shapes(self):
        return []
    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two 
        things: a list of tensors which the runtime will 
        evaluate, and a list of run-once ops"""
        # make the graph
        if not os.path.exists(args.dataset) and os.path.isfile(args.dataset):
            raise EnvironmentError("metadata file '{m}' either doesn't exist or is not a file".format(m=args.dataset))

        if in_queue is None:
            raise EnvironmentError("in queue is none")

        dataset_dir = os.path.dirname(args.dataset) 
        key = in_queue.dequeue()
        op = agd_mark_duplicates_local(key=key,
                                       local_directory=dataset_dir,
                                       outdir=dataset_dir, 
                                       parallel_parse=args.parse_parallel, 
                                       parallel_write=args.write_parallel)
        run_once = []
        return op, run_once 

markduplicate_service_ = MarkDuplicateService()

def service():
    return markduplicate_service_

def _make_writers(results_batch, output_dir):
    for column_handle, num_records, name, first_ord in results_batch:
        writer, first_o_passthru = persona_ops.agd_write_columns(record_id="fixme",
                                                       record_type=["results"],
                                                       column_handle=column_handle,
                                                       output_dir=output_dir + "/",
                                                       file_path=name,
                                                       first_ordinal=first_ord,
                                                       num_records=num_records,
                                                       name="agd_column_writer")
        yield writer # writes out the file path key (full path)


def agd_mark_duplicates_local(key, local_directory, outdir=None, parallel_parse=1, parallel_write=1):
    """
    key: tensor with chunk key string
    local_directory: the "base path" from which these should be read
    column_grouping_factor: the number of keys to put together
    parallel_parse: the parallelism for processing records (decomp)
    """

   
    parsed_results = tf.contrib.persona.persona_in_pipe(key=key, dataset_dir=local_directory, columns=["results"], parse_parallel=parallel_parse,
                                                        process_parallel=1)
   
    key, num_results, first_ord, result_handle = parsed_results[0]

    blp = persona_ops.buffer_list_pool(size=0, bound=False, name="local_read_buffer_pool")
    result_out = persona_ops.agd_mark_duplicates(results_handle=result_handle, num_records=num_results, 
            buffer_list_pool=blp, name="markdupsop")

    result_to_write = tf.contrib.persona.batch_pdq([result_out, num_results, key, first_ord],
                                        batch_size=1, num_dq_ops=parallel_write, name="to_write_queue")


    written = _make_writers(results_batch=result_to_write, output_dir=outdir)

    recs = [rec for rec in written]
    all_written_keys = tf.contrib.persona.batch_pdq(recs, num_dq_ops=1,
                                        batch_size=1, name="written_key_out")

    print("all written keys: {}".format(all_written_keys))
    return all_written_keys
  
  
