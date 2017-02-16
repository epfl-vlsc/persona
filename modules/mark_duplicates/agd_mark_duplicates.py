from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()

"""
ops specifically pertaining to agd mark duplicates

Contains convenience methods for creating common patterns for 
marking duplicates.

You may connect these yourself based on persona_ops (the parent module)
"""

def _key_maker(file_keys):
    num_file_keys = len(file_keys)

    string_producer = tf.train.string_input_producer(file_keys, num_epochs=1, shuffle=False)
    sp_output = string_producer.dequeue()

    return sp_output

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


def agd_mark_duplicates_local(file_keys, local_directory, outdir=None, parallel_parse=1, parallel_write=1):
    """
    file_keys: a list of Python strings of the file keys, which you should extract from the metadata file
    local_directory: the "base path" from which these should be read
    column_grouping_factor: the number of keys to put together
    parallel_parse: the parallelism for processing records (decomp)
    """

    key_producer = _key_maker(file_keys=file_keys)
   
    parsed_results = tf.contrib.persona.persona_in_pipe(key=key_producer, dataset_dir=local_directory, columns=["results"], parse_parallel=parallel_parse,
                                                        process_parallel=1)
   
    key, num_results, first_ord, result_handle = parsed_results[0]

    blp = persona_ops.buffer_list_pool(size=0, bound=False, name="local_read_buffer_pool")
    result_out = persona_ops.agd_mark_duplicates(results_handle=result_handle, num_records=num_results, 
            buffer_list_pool=blp, name="markdupsop")

    result_to_write = tf.train.batch_pdq([result_out, num_results, key, first_ord],
                                        batch_size=1, num_dq_ops=parallel_write, name="to_write_queue")


    written = _make_writers(results_batch=result_to_write, output_dir=outdir)

    recs = [rec for rec in written]
    all_written_keys = tf.train.batch_pdq(recs, num_dq_ops=1,
                                        batch_size=1, name="written_key_out")

    print("all written keys: {}".format(all_written_keys))
    return all_written_keys
  
  
