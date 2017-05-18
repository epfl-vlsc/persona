import os
from . import merge_sort
from ..common import service
from ..common.parse import numeric_min_checker

class CephSortSingleton(service.ServiceSingleton):
  class_type = merge_sort.CephSortService

class LocalSortSingleton(service.ServiceSingleton):
  class_type = merge_sort.LocalSortService

_singletons = [ CephSortSingleton(), LocalSortSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]


def get_tooltip():
  return "Sort an AGD dataset"




# ------------------------------------
def _run_local(args):
  meta_file = args.metadata_file
  if not os.path.exists(meta_file) and os.path.isfile(meta_file):
      raise EnvironmentError("metadata file '{m}' either doesn't exist or is not a file".format(m=meta_file))

  meta_file_dir = os.path.dirname(meta_file)
  if args.input is None:
      args.input = meta_file_dir
  if args.output is None:
      args.output = meta_file_dir

  metadata = {
      "sort_read_parallelism": args.sort_read_parallel,
      "column_grouping": args.column_grouping,
      "sort_parallelism": args.sort_parallel,
      "write_parallelism": args.write_parallel,
      "sort_process_parallelism": args.sort_process_parallel,
      "order_by": args.order_by
  }

  local_sort.run(args, {"params": metadata})

def _run_ceph(args):
  
  meta_file = args.metadata_file
  if not os.path.exists(meta_file) and os.path.isfile(meta_file):
      raise EnvironmentError("metadata file '{m}' either doesn't exist or is not a file".format(m=meta_file))

  ceph_params = args.ceph_params
  if not os.path.exists(ceph_params) and os.path.isfile(ceph_params):
      raise EnvironmentError("Ceph Params file '{}' isn't correct".format(ceph_params))

  a = args.ceph_read_chunk_size
  if a < 1:
      raise EnvironmentError("Ceph read chunk size most be strictly positive! Got {}".format(a))

  metadata = {
      "sort_read_parallelism": args.sort_read_parallel,
      "column_grouping": args.column_grouping,
      "sort_parallelism": args.sort_parallel,
      "write_parallelism_sort": args.write_parallel_sort,
      "write_parallelism_merge": args.write_parallel_merge,
      "sort_process_parallelism": args.sort_process_parallel,
      "order_by": args.order_by
  }

  ceph_sort.run(args, {"params": metadata})

def run(args):
  if args.storage == "local":
    _run_local(args)
  elif args.storage == "ceph":
    _run_ceph(args)
  else:
    raise Exception("Unknown storage subsystem to sort: {}".format(args.command))


def get_args(subparser):
  # args for ceph storage system
  cephsubparser = subsubparsers.add_parser(name="ceph", help="Options for sorting a dataset in Ceph object store")
  cephsubparser.add_argument("-r", "--sort-read-parallel", default=1, type=numeric_min_checker(minimum=1, message="read parallelism min for sort phase"),
                    help="total paralellism level for local read pipeline for sort phase")
  cephsubparser.add_argument("-c", "--column-grouping", default=5, help="grouping factor for parallel chunk sort",
                    type=numeric_min_checker(minimum=1, message="column grouping min"))
  cephsubparser.add_argument("-s", "--sort-parallel", default=1, help="number of sorting pipelines to run in parallel",
                    type=numeric_min_checker(minimum=1, message="sorting pipeline min"))
  cephsubparser.add_argument("-w", "--write-parallel-sort", default=1, help="number of ceph writers to use in parallel",
                    type=numeric_min_checker(minimum=1, message="writing pipeline min"))
  cephsubparser.add_argument("-x", "--write-parallel-merge", default=0, help="number of ceph writers to use in parallel",
                    type=numeric_min_checker(minimum=1, message="writing pipeline min"))
  cephsubparser.add_argument("--sort-process-parallel", default=1, type=numeric_min_checker(minimum=1, message="parallel processing for sort stage"),
                    help="parallel processing pipelines for sorting stage")
  cephsubparser.add_argument("-b", "--order-by", default="location", choices=["location", "metadata"], help="sort by this parameter [location | metadata]")
  cephsubparser.add_argument("--output-name", default="sorted", help="name for the output record")
  cephsubparser.add_argument("--chunk", default=2, type=numeric_min_checker(1, "need non-negative chunk size"), help="chunk size for final merge stage")
  cephsubparser.add_argument("--output-pool", default="", help="The Ceph cluster pool in which the output dataset should be written")
  cephsubparser.add_argument("--ceph-read-chunk-size", default=(2**26), type=int, help="minimum size to read from ceph storage, in bytes")
  cephsubparser.add_argument("ceph_params", help="Parameters for Ceph Reader")

