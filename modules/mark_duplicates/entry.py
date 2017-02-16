import os
import argparse
from . import local_mark_duplicates

def get_tooltip():
  return "Mark PCR duplicate reads in an aligned AGD dataset."

def run(args):
  meta_file = args.metadata_file
  if not os.path.exists(meta_file) and os.path.isfile(meta_file):
      raise EnvironmentError("metadata file '{m}' either doesn't exist or is not a file".format(m=meta_file))

  meta_file_dir = os.path.dirname(meta_file)
  if args.input is None:
      args.input = meta_file_dir

  local_mark_duplicates.run(args)

def get_args(subparser):
  def numeric_min_checker(minimum, message):
      def check_number(n):
          n = int(n)
          if n < minimum:
              raise argparse.ArgumentError("{msg}: got {got}, minimum is {minimum}".format(
                  msg=message, got=n, minimum=minimum
              ))
          return n
      return check_number

  default_dir_help = "Defaults to metadata_file's directory"
  subparser.add_argument("-p", "--parse-parallel", default=1, type=numeric_min_checker(minimum=1, message="read parallelism"),
                      help="total paralellism level for reading data from disk")
  subparser.add_argument("-w", "--write-parallel", default=1, help="number of writers to use",
                      type=numeric_min_checker(minimum=1, message="number of writers min"))
  subparser.add_argument("--input", help="input directory, containing all the files described in metadata_file\n{}".format(default_dir_help))
  subparser.add_argument("--logdir", default=".", help="Directory to write tensorflow summary data. Default is PWD")
  subparser.add_argument("metadata_file", help="the json metadata file describing the chunks in the original result set")

  

