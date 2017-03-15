import argparse
import multiprocessing
import os
from . import agd_output

def get_tooltip():
  return "Display AGD records on stdout"

def run(args):
  if not os.path.isabs(args.json_file):
    args.json_file = os.path.abspath(args.json_file)
  agd_output.run(args)

def get_args(subparser):

    subparser.add_argument("json_file", help="AGD dataset metadata")
    subparser.add_argument("start", type=int, help="The absolute index at which to start printing records")
    subparser.add_argument("finish", type=int, help="The absolute index at which to stop printing records")
    subparser.add_argument("-u", "--unpack", default=True, action='store_false', help="Whether or not to unpack binary bases")
  

