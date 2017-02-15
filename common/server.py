#!/usr/bin/env python3

# machine that you are distributing work to and change the task indices appropriately.
# Obtain the machine names from 'clusters.txt'
# Change the task index for each machine

import tensorflow as tf
import os
import sys
import socket
import json

import argparse

splitter = ":"

def get_args():
    def cluster_pair(arg):
        if splitter not in arg:
            raise EnvironmentError("Cluster Pair Argument '{}' is not properly formatted as 'host:port'".format(arg))
        a = arg.split(splitter)
        if len(a) != 2:
            raise EnvironmentError("Cluster Pair Argument '{}' is not properly formatted as 'host:port'".format(arg))
        try:
            host = a[0]
            port = a[1]
            return host, int(port)
        except ValueError as ve:
            raise EnvironmentError("Unable to parse second argument '{port}' of arg '{ar}'".format(port=a[1], ar=arg))

    parser = argparse.ArgumentParser(description="Starts the server, given the cluster spec",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--index", type=int, required=True, help="Index of this server")
    parser.add_argument("cluster_spec", nargs="+", type=cluster_pair, help="list of host:port for the cluster spec. host must be a DNS name")
    args = parser.parse_args()
    idx = args.index
    max_size = len(args.cluster_spec)
    if idx < 0 or idx >= max_size:
        raise EnvironmentError("Task index {idx} not valid for cluster spec size {sz}".format(idx=idx, sz=max_size))
    task_idx = args.cluster_spec[idx]
    task_host = task_idx[0]
    this_host = socket.gethostname()
    if not task_host.startswith(this_host):
        raise EnvironmentError("task index host '{idx}' does not start with '{actual}".format(idx=task_host, actual=this_host))
    return args

def run(cluster_def, task_index):
    cluster_spec = { "persona" : ["{host}:{port}".format(host=a[0], port=a[1]) for a in cluster_def] }
    cluster = tf.train.ClusterSpec(cluster_spec)
    server = tf.train.Server(cluster, job_name="persona", task_index=task_index)
    server.join()

if __name__ == "__main__":
    args = get_args()
    run(cluster_def=args.cluster_spec, task_index=args.index)
