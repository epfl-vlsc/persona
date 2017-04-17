import argparse
import multiprocessing
import os
import getpass
from . import snap_align

def get_tooltip():
  return "Perform single or paired-end alignment on an AGD dataset using SNAP"

def get_service():
  return snap_align.service()

def get_args(subparser):

    subparser.add_argument("-p", "--parallel", type=int, default=2, help="parallel decompression")
    subparser.add_argument("-e", "--enqueue", type=int, default=1, help="parallel enqueuing")
    subparser.add_argument("-m", "--mmap-queue", type=int, default=2, help="size of the mmaped file record queue")
    subparser.add_argument("-a", "--aligners", type=int, default=1, help="number of aligners")
    subparser.add_argument("-t", "--aligner-threads", type=int, default=multiprocessing.cpu_count(), help="the number of threads to use per aligner")
    subparser.add_argument("-x", "--subchunking", type=int, default=5000, help="the size of each subchunk (in number of reads)")
    subparser.add_argument("-w", "--writers", type=int, default=0, help="the number of writer pipelines")
    subparser.add_argument("-c", "--compress", default=False, action='store_true', help="compress the output")
    subparser.add_argument("-s", "--max-secondary", type=int, default=0, help="Max secondary results to store. >= 0 ")
    subparser.add_argument("--deep-verify", default=False, action='store_true', help="verify record integrity")
    subparser.add_argument("--paired", default=False, action='store_true', help="interpret dataset as interleaved paired dataset")
    subparser.add_argument("--ceph-cluster-name", default="ceph", help="name for the ceph cluster")
    subparser.add_argument("--ceph-user-name", default="client.dcsl1024", help="ceph username")
    subparser.add_argument("--ceph-pool-name", default="chunk_100000", help="ceph pool name")
    subparser.add_argument("--ceph-conf-path", default=os.path.join(os.path.dirname(__file__), "ceph_agd_align/ceph.conf"), help="path for the ceph configuration")
    subparser.add_argument("--local-path", help="if set, causes the cluster machines to read from this path on their local disks (NOT THE NETWORK)")
    subparser.add_argument("-i", "--index-path", default="/scratch/stuart/ref_index", help="location of the ref index on all machines. Make sure all machines have this path!")
    subparser.add_argument("--null", type=float, help="use the null aligner instead of actually aligning")
    subparser.add_argument("--ceph-read-chunk-size", default=(2**26), type=int, help="minimum size to read from ceph storage, in bytes")
    subparser.add_argument("--summary", default=False, action="store_true", help="Add TensorFlow summary info to the graph")
    subparser.add_argument("-r", "--args", default="", help="A string containing SNAP-specific args. Enclose in \" \" for multiple args")

  

