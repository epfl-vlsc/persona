#!/usr/bin/env python3

import importlib
import os
import argparse
import sys
import json
import tensorflow as tf
import shutil

def setup_output_dir(dirname="cluster_traces"):
    trace_path = os.path.join(os.path.dirname(os.getcwd()), dirname)
    if os.path.exists(trace_path):
        # nuke it
        shutil.rmtree(trace_path)
    os.makedirs(trace_path)
    return trace_path

def execute(args, modules):
  if not args.modes == 'local':
    raise Exception("Local runtime received args without local mode")
  module = modules[args.local]

  service = module.get_service()

  # args ensure every module must specify dataset
  with open(args.dataset, 'r') as j:
    dataset = json.load(j)
  
  records = dataset['records']
  chunknames = []
  for record in records:
    chunknames.append(record['path'])

  producer = tf.train.string_input_producer(string_tensor=chunknames, num_epochs=1, 
      shuffle=False, name="executor_input_producer")

  print(producer)
  service_ops, service_init_ops = service.make_graph(producer, args)

  init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer()]
  init_ops.extend(service_init_ops)

  # service graph may have summary nodes
  merged = tf.summary.merge_all()
  summary = True if hasattr(args, 'summary') else False

  with tf.Session() as sess:
      if summary:
          trace_dir = setup_output_dir(dirname=args.local + "_summary")
          service_ops.append(merged)
          summary_writer = tf.summary.FileWriter(trace_dir, graph=sess.graph, max_queue=2**20, flush_secs=10**4)
          count = 0

      sess.run(init_ops)

      coord = tf.train.Coordinator()
      print("Local executor starting {} ...".format(args.local))
      threads = tf.train.start_queue_runners(coord=coord, sess=sess)
      while not coord.should_stop():
          try:
              a = sess.run(service_ops)
              if summary:
                  summary_writer.add_summary(a[-1], global_step=count)
                  count += 1
          except tf.errors.OutOfRangeError as e:
              print('Got out of range error!')
              break
      print("Coord requesting stop")
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
