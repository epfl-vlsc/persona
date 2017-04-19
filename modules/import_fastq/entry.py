from . import convert
from ..common.parse import numeric_min_checker

def get_tooltip():
  return "Import FASTQ files into an AGD dataset"

def get_service():
  return convert.service()

def get_services():
  return []

def get_args(subparser):
  subparser.add_argument("-c", "--chunk", type=numeric_min_checker(1, "chunk size"), default=10000, help="chunk size to create records")
  subparser.add_argument("-p", "--parallel-conversion", type=numeric_min_checker(1, "parallel conversion"), default=1, help="number of parallel converters")
  subparser.add_argument("-n", "--name", required=True, help="name for the record")
  #subparser.add_argument("-o", "--out", default=".", help="directory to write the final record to")
  subparser.add_argument("-w", "--write", default=1, type=numeric_min_checker(1, "write parallelism"), help="number of parallel writers")
  subparser.add_argument("--summary", default=False, action='store_true', help="run with tensorflow summary nodes")
  subparser.add_argument("--compress", default=False, action='store_true', help="compress output blocks")
  subparser.add_argument("--compress-parallel", default=1, type=numeric_min_checker(1, "compress parallelism"), help="number of parallel compression pipelines")
  subparser.add_argument("fastq_files", nargs="+", help="the fastq file to convert")
