import argparse
import multiprocessing
import os
import json
import tensorflow as tf
from ..common.service import Service

persona_ops = tf.contrib.persona.persona_ops()

class BamExportService(Service):
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

        key = in_queue.dequeue()
        ops, run_once = run(key, args)

        return ops, run_once

bamexport_service_ = BamExportService()

def service():
    return bamexport_service_


def export_bam(key, args):
  with open(args.dataset, 'r') as j:
    manifest = json.load(j)

  if 'reference' not in manifest:
    raise Exception("No reference data in manifest {}. Unaligned BAM not yet supported. Please align dataset first.".format(args.dataset))

  bp_handle = persona_ops.buffer_pool(size=10, bound=False, name="buf_pool")
  mmap_pool = persona_ops.m_map_pool(size=10,  bound=False, name="file_mmap_buffer_pool")
  
  local_dir = os.path.dirname(args.dataset)
  parsed_chunks = tf.contrib.persona.persona_in_pipe(dataset_dir=local_dir, columns=["results", "base", "qual", "meta"], key=key, 
                                                     mmap_pool=mmap_pool, buffer_pool=pp)
  key, num_recs, first_ord, results, bases, quals, meta = parsed_chunks

  if args.output_path == "":
    output_path = manifest['name'] + ".bam"
  else:
    output_path = args.output_path

  ref_lens = []
  ref_seqs = []

  for ref_seq, ref_len in manifest['reference']:
    ref_lens.append(ref_len)
    ref_seqs.append(ref_seq)

  pg_id = "personaAGD" # TODO get from manifest
  read_group = manifest['name'] 
  agd_to_bam = persona_ops.agd_output_bam(results_handle=results, bases_handle=bases, 
                                          qualities_handle=quals, metadata_handle=meta,
                                          num_records=num_recs, path=output_path,
                                          ref_sequences=ref_seqs, ref_seq_sizes=ref_lens,
                                          pg_id=pg_id, read_group=read_group, sort_order='unsorted',
                                          num_threads=args.threads)
  
  return [agd_to_bam], []

