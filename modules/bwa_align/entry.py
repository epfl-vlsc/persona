import argparse
import multiprocessing
import os

def run(args):
  if args.null is not None:
      if args.null < 0.0:
          raise EnvironmentError("null wait time must be strictly non-negative. Got {}".format(args.null))

  index_path = args.index_path
  if not (os.path.exists(index_path) and os.path.isfile(index_path)):
      raise EnvironmentError("Index path '{}' specified incorrectly. Should be path/to/index.fa".format(index_path))

  if args.finalizer_threads == 0 and args.paired:
      args.finalizer_threads = int(args.aligner_threads*0.31)
      args.aligner_threads = args.aligner_threads - args.finalizer_threads
  else:
      if args.aligner_threads + args.finalizer_threads > multiprocessing.cpu_count():
          raise EnvironmentError("More threads than available on machine {}".format(args.aligner_threads + args.finalizer_threads))

  print("aligner {} finalizer {}".format(args.aligner_threads, args.finalizer_threads))

  local_path = args.local_path
  if not (os.path.exists(local_path) and os.path.isdir(local_path)):
      raise EnvironmentError("Local path'{l}' not found".format(l=local_path))

  if args.writers < 0:
      raise EnvironmentError("need a strictly positive number of writers, got {}".format(args.writers))
  if args.parallel < 1:
      raise EnvironmentError("need at least 1 parallel dequeue, got {}".format(args.parallel))
  if args.enqueue < 1:
      raise EnvironmentError("need at least 1 parallel enqueue, got {}".format(args.enqueue))
  print("Running bwa align!")

def get_args(subparser):
  subparser.add_argument("-p", "--parallel", type=int, default=2, help="parallel decompression")
  subparser.add_argument("-e", "--enqueue", type=int, default=1, help="parallel enqueuing")
  subparser.add_argument("-m", "--mmap-queue", type=int, default=2, help="size of the mmaped file record queue")
  subparser.add_argument("-t", "--aligner-threads", type=int, default=multiprocessing.cpu_count(), help="the number of threads to use for BWA first state")
  subparser.add_argument("-f", "--finalizer-threads", type=int, default=0, help="the number of threads to use for BWA second stage")
  subparser.add_argument("-x", "--subchunking", type=int, default=5000, help="the size of each subchunk (in number of reads)")
  subparser.add_argument("-w", "--writers", type=int, default=0, help="the number of writer pipelines")
  subparser.add_argument("-c", "--compress", default=False, action='store_true', help="compress the output")
  subparser.add_argument("-i", "--index-path", default="/scratch/bwa_index/hs38DH.fa")
  subparser.add_argument("--paired", default=False, action='store_true', help="interpret dataset as interleaved paired dataset")
  subparser.add_argument("--null", type=float, required=False, help="use the null aligner instead of actually aligning")
  subparser.add_argument("--deep-verify", default=False, action='store_true', help="verify record integrity")
  # TODO this is rigid, needs to be changed to get from the queue service!
  subparser.add_argument("--summary", default=False, action="store_true", help="Add TensorFlow summary info to the graph")
  subparser.add_argument("local_path", help="Read from this path on their local disks (NOT THE NETWORK)")
  subparser.add_argument("metadata_file", help="the json metadata file describing the chunks in the original result set")

  

