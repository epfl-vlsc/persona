import argparse
import multiprocessing
import os
from . import snap_align_local

def get_tooltip():
  return "Perform single or paired-end alignment on an AGD dataset using SNAP"

def run(args):
  ceph_conf_path= args.ceph_conf_path
  if not os.path.exists(ceph_conf_path) and os.path.isfile(ceph_conf_path):
      raise EnvironmentError("Ceph conf path {} either isn't a file, or doesn't exist".format(ceph_conf_path))
  args.ceph_conf_path = os.path.abspath(ceph_conf_path)

  a = args.ceph_read_chunk_size
  if a < 1:
      raise EnvironmentError("Ceph read chunk size most be strictly positive! Got {}".format(a))

  if args.aligners < 1:
      raise EnvironmentError("Must have a strictly positive number of aligners! Got {}".format(args.aligners))

  if args.null is not None:
      if args.null < 0.0:
          raise EnvironmentError("null wait time must be strictly non-negative. Got {}".format(args.null))

  index_path = args.index_path
  if not (os.path.exists(index_path) and os.path.isdir(index_path)):
      raise EnvironmentError("Index path '{}' specified incorrectly. Doesn't exist".format(index_path))

  if args.aligner_threads < args.aligners:
      raise EnvironmentError("Aligner must have at least 1 degree of parallelism! Got {}".format(args.aligner_threads))

  local_path = args.local_path
  if local_path is not None:
      if not (os.path.exists(local_path) and os.path.isdir(local_path)):
          raise EnvironmentError("Local path'{l}' not found".format(l=local_path))

  if args.writers < 0:
      raise EnvironmentError("need a strictly positive number of writers, got {}".format(args.writers))
  if args.parallel < 1:
      raise EnvironmentError("need at least 1 parallel dequeue, got {}".format(args.parallel))
  if args.enqueue < 1:
      raise EnvironmentError("need at least 1 parallel enqueue, got {}".format(args.enqueue))

  snap_align_local.run(args):

def get_args(subparser):
  subparser.add_argument("-p", "--parallel", type=int, default=2, help="parallel decompression")
  subparser.add_argument("-e", "--enqueue", type=int, default=1, help="parallel enqueuing")
  subparser.add_argument("-m", "--mmap-queue", type=int, default=2, help="size of the mmaped file record queue")
  subparser.add_argument("-a", "--aligners", type=int, default=1, help="number of aligners")
  subparser.add_argument("-t", "--aligner-threads", type=int, default=multiprocessing.cpu_count(), help="the number of threads to use per aligner")
  subparser.add_argument("-x", "--subchunking", type=int, default=5000, help="the size of each subchunk (in number of reads)")
  subparser.add_argument("-w", "--writers", type=int, default=0, help="the number of writer pipelines")
  subparser.add_argument("-c", "--compress", default=False, action='store_true', help="compress the output")
  subparser.add_argument("-i", "--index-path", default="/scratch/stuart/ref_index")
  subparser.add_argument("--paired", default=False, action='store_true', help="interpret dataset as interleaved paired dataset")
  subparser.add_argument("--null", type=float, required=False, help="use the null aligner instead of actually aligning")
  subparser.add_argument("--deep-verify", default=False, action='store_true', help="verify record integrity")
  subparser.add_argument("--local-path", default=None, help="if set, causes the cluster machines to read from this path on their local disks (NOT THE NETWORK)")
  subparser.add_argument("--ceph-read-chunk-size", default=(2**26), type=int, help="minimum size to read from ceph storage, in bytes")
  subparser.add_argument("--ceph-cluster-name", default="ceph", help="name for the ceph cluster")
  subparser.add_argument("--ceph-user-name", default="client.dcsl1024", help="ceph username")
  # TODO this is rigid, needs to be changed to get from the queue service!
  subparser.add_argument("--ceph-pool-name", default="chunk_100000", help="ceph pool name")
  subparser.add_argument("--ceph-conf-path", default=os.path.join(os.path.dirname(__file__), "ceph_agd_align/ceph.conf"), help="path for the ceph configuration")
  subparser.add_argument("--summary", default=False, action="store_true", help="Add TensorFlow summary info to the graph")
  subparser.add_argument("--queue-host", help="queue service host")
  subparser.add_argument("--upstream-pull-port", type=int, default=5556, help="port to request new work items on")
  subparser.add_argument("--downstream-push-port", type=int, default=5557, help="port to send completed work items on")

  

