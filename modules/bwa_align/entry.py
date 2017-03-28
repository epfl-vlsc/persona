import argparse
import multiprocessing
import os
from . import persona_bwa
from ..common import parse

def get_service():
  return persona_bwa.service()

def get_tooltip():
  return "Perform single or paired-end alignment on an AGD dataset using BWA"

def get_graph_args(subparser):
  subparser.add_argument("-p", "--parallel", type=int, default=2, help="parallel decompression")
  subparser.add_argument("-e", "--enqueue", type=int, default=1, help="parallel enqueuing")
  subparser.add_argument("-m", "--mmap-queue", type=int, default=2, help="size of the mmaped file record queue")
  subparser.add_argument("-t", "--aligner-threads", type=int, default=multiprocessing.cpu_count(), help="the number of threads to use for BWA first state")
  subparser.add_argument("-f", "--finalizer-threads", type=int, default=0, help="the number of threads to use for BWA second stage")
  subparser.add_argument("-x", "--subchunking", type=int, default=5000, help="the size of each subchunk (in number of reads)")
  subparser.add_argument("-w", "--writers", type=int, default=0, help="the number of writer pipelines")
  subparser.add_argument("-c", "--compress", default=False, action='store_true', help="compress the output")
  subparser.add_argument("-i", "--index-path", default="/scratch/bwa_index/hs38DH.fa")
  subparser.add_argument("-s", "--max-secondary", default=1, help="Max secondary results to store. >= 1 (required for chimaric results")
  subparser.add_argument("--paired", default=False, action='store_true', help="interpret dataset as interleaved paired dataset")
  subparser.add_argument("--null", type=float, required=False, help="use the null aligner instead of actually aligning")
  subparser.add_argument("--deep-verify", default=False, action='store_true', help="verify record integrity")
  # TODO this is rigid, needs to be changed to get from the queue service!
  subparser.add_argument("--summary", default=False, action="store_true", help="Add TensorFlow summary info to the graph")
  #subparser.add_argument("local_path", help="Read from this path on their local disks (NOT THE NETWORK)")

def get_run_args(subparser):
  parse.add_dataset(subparser)
