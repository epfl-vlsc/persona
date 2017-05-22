#!/usr/bin/env python3

import argparse
import subprocess
import itertools
import os

def get_args():
    parser = argparse.ArgumentParser(description="run some basic experiments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--enqueue", type=int, nargs="+", help="parallel reading")
    parser.add_argument("-p", "--parallel", type=int, nargs="+", help="parallel processing")
    parser.add_argument("-a", "--aligners", type=int, nargs="+", help="parallel aligners")
    parser.add_argument("-x", "--subchunking", type=int, nargs="+", help="subchunking level")
    parser.add_argument("-c", "--compress-parallel", type=int, nargs="+", help="number of parallel compressors")
    parser.add_argument("-d", "--dataset-dir", required=True, help="dataset local directory")
    parser.add_argument("-w", "--writers", type=int, nargs="+", help="number of writers")
    parser.add_argument("-t", "--threads", type=int, nargs="+", help="number of aligner threads")
    parser.add_argument("-r", "--record-dir", required=True, help="directory in which to record results")
    parser.add_argument("-i", "--iterations", type=int, help="limit number of iterations")
    parser.add_argument("--summary", default=False, action='store_true', help="add summary")
    parser.add_argument("dataset", help="path to the json file for the dataset")
    args = parser.parse_args()
    args.executable = os.path.join(os.path.abspath(os.path.dirname(__file__)), "persona")
    return args

def run_experiment(enqueue, parallel, aligner_count, subchunking, compression, writer_count, aligner_threads, executable, dataset_dir, dataset, record_dir, iterations, summary):
    command_line = "{exec} snap_align local -p {parallel} -e {enqueue} {summary} " \
                   "-a {aligners} -t {threads} -w {writers} -c --compress-parallel {comp} " \
                   "--record-directory {rec_dir} {iterations} -x {subchunk} " \
                   "-d {dataset_dir} {dataset}".format(exec=executable,
                                                       parallel=parallel,
                                                       aligners=aligner_count,
                                                       threads=aligner_threads,
                                                       comp=compression,
                                                       summary="--summary" if summary else "",
                                                       enqueue=enqueue,
                                                       writers=writer_count,
                                                       subchunk=subchunking,
                                                       dataset_dir=dataset_dir,
                                                       dataset=dataset,
                                                       iterations="--iterations {}".format(iterations) if iterations is not None else "",
                                                       rec_dir=record_dir)
    print("Running command line:\n{}".format(command_line))
    return
    a = os.environ.copy()
    a["TF_CPP_MIN_LOG_LEVEL"] = "1"
    subprocess.run(command_line, shell=True, check=True, env=a)

def run(args):
    for enqueue, parallel, aligners, subchunking, compress_parallel, writers, threads in itertools.product(args.enqueue, args.parallel, args.aligners, args.subchunking, args.compress_parallel,
                      args.writers, args.threads):
        run_experiment(enqueue=enqueue,
                       parallel=parallel,
                       aligner_count=aligners,
                       compression=compress_parallel,
                       writer_count=writers,
                       aligner_threads=threads,
                       subchunking=subchunking,
                       executable=args.executable,
                       dataset=args.dataset,
                       record_dir=args.record_dir,
                       dataset_dir=args.dataset_dir,
                       iterations=args.iterations,
                       summary=args.summary)

if __name__ == "__main__":
    run(get_args())
