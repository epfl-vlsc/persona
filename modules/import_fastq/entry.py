import argparse
import multiprocessing
import os

def get_tooltip():
  return "Import FASTQ files into an AGD dataset"

def run(args):
  if len(args.name) > 31:
      parser.error("Name must be at most 31 characters. Got {}".format(len(args.name)))
  args.fastq_files = [make_abs(p, os.path.isfile) for p in args.fastq_files]
  args.out = os.path.join(make_abs(args.out, os.path.isdir), args.name)
  if os.path.exists(args.out):
      subprocess.run("rm -rf {}".format(args.out), shell=True, check=True)
  os.makedirs(args.out)
  args.logdir = make_abs(args.logdir, os.path.isdir)
  args.compress = False # force false for now
  print("Running import fastq!")

def get_args(subparser):
  def numeric_min(min):
      def check(a):
          a = int(a)
          if a < min:
              raise argparse.ArgumentError("Value must be at least {min}. got {actual}".format(min=min, actual=a))
          return a
      return check

  def make_abs(path, check_func):
      if not (os.path.exists(path) and check_func(path)):
          parser.error("'{}' is not a valid path".format(path))
          return
      return os.path.abspath(path)

  subparser.add_argument("-c", "--chunk", type=numeric_min(1), default=10000, help="chunk size to create records")
  subparser.add_argument("-p", "--parallel-conversion", type=numeric_min(1), default=1, help="number of parallel converters")
  subparser.add_argument("-n", "--name", required=True, help="name for the record")
  subparser.add_argument("-o", "--out", default=".", help="directory to write the final record to")
  subparser.add_argument("--logdir", default=".", help="Directory to write tensorflow summary data. Default is PWD")
  subparser.add_argument("-w", "--write", default=1, type=numeric_min(1), help="number of parallel writers")
  subparser.add_argument("--summary", default=False, action='store_true', help="run with tensorflow summary nodes")
  subparser.add_argument("--compress", default=False, action='store_true', help="compress output blocks")
  subparser.add_argument("--compress-parallel", default=1, type=numeric_min(1), help="number of parallel compression pipelines")
  subparser.add_argument("fastq_files", nargs="+", help="the fastq file to convert")

  

