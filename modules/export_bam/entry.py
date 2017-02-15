import argparse
import multiprocessing
import os
import json
from . import export_bam

def get_tooltip():
  return "Export AGD dataset to BAM"

def run(args):
  export_bam.export_bam(args)

def get_args(subparser):

  subparser.add_argument("json_file", help="AGD dataset metadata file")
  subparser.add_argument("-p", "--parallel-parse", default=2, help="Parallelism of decompress stage")
  subparser.add_argument("-o", "--output-path", default="", help="Output bam file path")
  subparser.add_argument("-t", "--threads", type=int, default=multiprocessing.cpu_count(), 
      help="Number of threads to use for compression [{}]".format(multiprocessing.cpu_count()))
  

