import argparse
import multiprocessing
import os
import getpass
from . import snap_align

def get_tooltip():
  return "Perform single or paired-end alignment on an AGD dataset using SNAP"

def run(args):
    if not os.path.isabs(args.json_file):
        args.json_file = os.path.abspath(args.json_file)

    if args.local_tensorflow is not None:
        if not os.path.exists(args.local_tensorflow) and os.path.isdir(args.local_tensorflow):
            raise Exception("Local TensorFlow repository not found at: {}".format(args.local_tensorflow))

    if args.distribute < 1:
        raise Exception("Need to distribute on at least one node. Got {}".format(args.distribute))

    if args.null is not None:
        if args.null < 0.0:
            raise EnvironmentError("null wait time must be strictly non-negative. Got {}".format(args.null))

    a = args.ceph_read_chunk_size
    if a < 1:
        raise EnvironmentError("Ceph read chunk size most be strictly positive! Got {}".format(a))

    if args.aligners < 1:
        raise EnvironmentError("Must have a strictly positive number of aligners! Got {}".format(args.aligners))

    if args.aligner_threads < args.aligners:
        raise EnvironmentError("Aligner must have at least 1 degree of parallelism! Got {}".format(args.aligner_threads))

    if args.writers < 0:
        raise EnvironmentError("need a strictly positive number of writers, got {}".format(args.writers))
    if args.parallel < 1:
        raise EnvironmentError("need at least 1 parallel dequeue, got {}".format(args.parallel))
    if args.enqueue < 1:
        raise EnvironmentError("need at least 1 parallel enqueue, got {}".format(args.enqueue))

    if not os.path.exists(args.json_file):
        raise EnvironmentError("File not found at path {path}".format(path=args.json_file))

    snap_align.run(args)

def get_args(subparser):

    subparser.add_argument("-p", "--parallel", type=int, default=2, help="parallel decompression")
    subparser.add_argument("-e", "--enqueue", type=int, default=1, help="parallel enqueuing")
    subparser.add_argument("-m", "--mmap-queue", type=int, default=2, help="size of the mmaped file record queue")
    subparser.add_argument("-a", "--aligners", type=int, default=1, help="number of aligners")
    subparser.add_argument("-t", "--aligner-threads", type=int, default=multiprocessing.cpu_count(), help="the number of threads to use per aligner")
    subparser.add_argument("-x", "--subchunking", type=int, default=5000, help="the size of each subchunk (in number of reads)")
    subparser.add_argument("-w", "--writers", type=int, default=0, help="the number of writer pipelines")
    subparser.add_argument("-c", "--compress", default=False, action='store_true', help="compress the output")
    #subparser.add_argument("-d", "--distribute", type=int, default=1, help="Specify the number of distributed machines")
    subparser.add_argument("--paired", default=False, action='store_true', help="interpret dataset as interleaved paired dataset")
    subparser.add_argument("--ceph-user", default="client.dcsl1024", help="ceph username")
    subparser.add_argument("--ceph-cluster", default="ceph", help="name for ceph cluster")
    subparser.add_argument("--unload-master", default=False, action="store_true", help="do not place the alignment graph on the master server")
    subparser.add_argument("--remote-path", default="/work/{username}".format(username=getpass.getuser()),
                        help="the remote path that contains tensorflow and tf-align repos\nAssumes the same for all machines!")
    subparser.add_argument("--local-tensorflow", help="local path to tensorflow repository to check version number")
    subparser.add_argument("--local-path", help="if set, causes the cluster machines to read from this path on their local disks (NOT THE NETWORK)")
    subparser.add_argument("-i", "--index-path", default="/scratch/stuart/ref_index", help="location of the ref index on all machines. Make sure all machines have this path!")
    subparser.add_argument("--null", type=float, help="use the null aligner instead of actually aligning")
    subparser.add_argument("--ceph-read-chunk-size", default=(2**26), type=int, help="minimum size to read from ceph storage, in bytes")
    subparser.add_argument("--summary", default=False, action="store_true", help="Add TensorFlow summary info to the graph")
    subparser.add_argument("json_file", help="An AGD dataset metadata file")
    subparser.add_argument("hosts", nargs="+", help="names for hosts to run on")

  

