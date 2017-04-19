#!/usr/bin/env python3
import tensorflow as tf
import argparse
import os
import json
from ..common.service import Service

persona_ops = tf.contrib.persona.persona_ops()

class DisplayService(Service):
   
    #default inputs
    def get_shortname(self):
        return "display"

    def add_graph_args(self, parser):
        parser.add_argument("start", type=int, help="The absolute index at which to start printing records")
        parser.add_argument("finish", type=int, help="The absolute index at which to stop printing records")
        parser.add_argument("-u", "--unpack", default=True, action='store_false', help="Whether or not to unpack binary bases")

    def distributed_capability(self):
        return False

    def output_dtypes(self):
        return []
    def output_shapes(self):
        return []

    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two 
        things: a list of tensors which the runtime will 
        evaluate, and a list of run-once ops"""
        # make the graph

        run_once = run(args)

        return [], run_once

def run(args):

  dataset_params = args.dataset
  records = dataset_params['records']
  first_record = records[0]
  chunk_size = first_record["last"] - first_record["first"]
  chunknames = []
  for record in records:
    chunknames.append(record['path'])

  if (args.finish <= args.start):
    args.finish = args.start + 1


  pathname = os.path.dirname(args.dataset) + '/'

  path = tf.constant(pathname)
  start = tf.constant(args.start, dtype=tf.int32)
  finish = tf.constant(args.finish, dtype=tf.int32)
  names = tf.constant(chunknames)
  size = tf.constant(chunk_size)
  output = persona_ops.agd_output(path=path, chunk_names=names, chunk_size=size, 
      start=start, finish=finish, columns=['metadata', 'base', 'qual', 'results', 'secondary0'])

  return [output]

