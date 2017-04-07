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
  if args.modes != 'local':
    raise Exception("Local runtime received args without local mode")
  module = modules[args.local]

  service = module.get_service()
  run_arguments = service.extract_run_args(args=args)
  input_dtypes = service.input_dtypes()
  input_shapes = service.input_shapes()

  # We need the batch_join to "close" with a stop exception after enqueuing once
  # and the FIFOQueue so the graph can decide on its own how much parallelism it wants
  first_in_queue = tf.train.batch_join(tensors_list=run_arguments, capacity=len(run_arguments),
                                       batch_size=1)
  in_queue = tf.FIFOQueue(dtypes=input_dtypes, # [a.dtype for a in first_in_queue],
                          shapes=input_shapes, # [a.get_shape() for a in first_in_queue],
                          capacity=len(run_arguments))
  tf.train.add_queue_runner(
      tf.train.QueueRunner(enqueue_ops=in_queue.enqueue(first_in_queue))
  )

  # TODO currently we assume all the service_ops are the same
  service_ops, service_init_ops = service.make_graph(in_queue=in_queue,
                                                     args=args)
  assert len(service_ops) + len(service_init_ops) > 0

  init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer()]

  # service graph may have summary nodes
  merged = tf.summary.merge_all()
  summary = args.summary if hasattr(args, 'summary') else False

  results = []
  with tf.Session() as sess:
      if summary:
          trace_dir = setup_output_dir(dirname=args.local + "_summary")
          service_ops.append(merged)
          summary_writer = tf.summary.FileWriter(trace_dir, graph=sess.graph, max_queue=2**20, flush_secs=10**4)
          count = 0

      sess.run(init_ops)
      if len(service_init_ops) > 0:
          sess.run(service_init_ops)

      # its possible the service is a simple run once
      if len(service_ops) > 0:
          service_sink = tf.train.batch_join(tensors_list=service_ops, batch_size=1)
          coord = tf.train.Coordinator()
          print("Local executor starting {} ...".format(args.local))
          threads = tf.train.start_queue_runners(coord=coord, sess=sess)
          while not coord.should_stop():
              try:
                  result = sess.run(service_sink)
                  if summary:
                      results.append(result[:-1])
                      summary_writer.add_summary(result[-1], global_step=count)
                      count += 1
                  else:
                      results.append(result)
              except tf.errors.OutOfRangeError:
                  print('Got out of range error!')
                  break
          print("Coord requesting stop")
          coord.request_stop()
          coord.join(threads, stop_grace_period_secs=10)

      # service.on_finish(results)
