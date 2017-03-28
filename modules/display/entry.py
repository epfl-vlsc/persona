import argparse
import multiprocessing
import os
from . import agd_output
from ..common import parse

def get_tooltip():
  return "Display AGD records on stdout"

def get_service():
  return agd_output.service()

def get_graph_args(subparser):
    subparser.add_argument("start", type=int, help="The absolute index at which to start printing records")
    subparser.add_argument("finish", type=int, help="The absolute index at which to stop printing records")
    subparser.add_argument("-u", "--unpack", default=True, action='store_false', help="Whether or not to unpack binary bases")

def get_run_args(subparser):
  parse.add_dataset(subparser)
