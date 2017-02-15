import argparse
import multiprocessing
import os
import json
import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()

def get_tooltip():
  return "Export AGD dataset to BAM"

def run(args):
  with open(args.json_file, 'r') as j:
    manifest = json.load(j)

  if 'reference' not in manifest:
    raise Exception("No reference data in manifest {}. Unaligned BAM not yet supported. Please align dataset first.".format(args.json_file))

  bp_handle = persona_ops.buffer_pool(size=10, bound=False, name="buf_pool")
  mmap_pool = persona_ops.m_map_pool(size=10,  bound=False, name="file_mmap_buffer_pool")
  
  local_dir = os.path.dirname(args.json_file)
  parsed_chunks = tf.contrib.persona.persona_in_pipe(metadata_path=args.json_file, dataset_dir=local_dir, columns=["results", "base", "qual", "meta"], key=None, 
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
  read_group = "thereadgroup" # TODO get from manifest
  agd_to_bam = persona_ops.agd_output_bam(results_handle=results, bases_handle=bases, 
                                          qualities_handle=quals, metadata_handle=meta,
                                          num_records=num_recs, path=output_path,
                                          ref_sequences=ref_seqs, ref_seq_sizes=ref_lens,
                                          pg_id=pg_id, read_group=read_group, sort_order='unsorted',
                                          num_threads=args.threads)

    
  init_op = [tf.local_variables_initializer(), tf.global_variables_initializer()]

  #print(os.getpid())
  #import ipdb; ipdb.set_trace()
  with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    print("Starting Run")
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    count = 0
    while not coord.should_stop():
        try:
            sess.run([agd_to_bam])
            print("round: {}".format(count)); count += 1    
        except tf.errors.OutOfRangeError:
            print('Got out of range error!')
            break
    print("Coord requesting stop")
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
  print("Running export BAM!")

def get_args(subparser):

  subparser.add_argument("json_file", help="AGD dataset metadata file")
  subparser.add_argument("-p", "--parallel-parse", default=2, help="Parallelism of decompress stage")
  subparser.add_argument("-o", "--output-path", default="", help="Output bam file path")
  subparser.add_argument("-t", "--threads", type=int, default=multiprocessing.cpu_count(), 
      help="Number of threads to use for compression [{}]".format(multiprocessing.cpu_count()))
  

