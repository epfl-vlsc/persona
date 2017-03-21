#!/usr/bin/env python3
import tensorflow as tf
import argparse
import os
import json
from ..common.service import Service

persona_ops = tf.contrib.persona.persona_ops()

class DisplayService(Service):
   
    #default inputs

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

display_service_ = DisplayService()

def service():
    return display_service_

def run(args):

  with open(args.dataset, 'r') as j:
    dataset_params = json.load(j)

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

