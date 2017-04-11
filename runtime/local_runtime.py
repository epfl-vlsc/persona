import os
import tensorflow as tf
import shutil
from tensorflow.contrib.persona import pipeline

def setup_output_dir(dirname="cluster_traces"):
    trace_path = os.path.join(os.path.dirname(os.getcwd()), dirname)
    if os.path.exists(trace_path):
        # nuke it
        shutil.rmtree(trace_path)
    os.makedirs(trace_path)
    return trace_path

def execute(args, modules):
  if args.mode != 'local':
    raise Exception("Local runtime received args without local mode")
  module = modules[args.local]

  service_mode = args.service
  service = module.lookup_service(name=service_mode)
  run_arguments = tuple(service.extract_run_args(args=args))
  input_dtypes = service.input_dtypes()
  input_shapes = service.input_shapes()

  # We need the batch_join to "close" with a stop exception after enqueuing once
  # and the FIFOQueue so the graph can decide on its own how much parallelism it wants
  in_queue = tf.train.input_producer(input_tensor=run_arguments, num_epochs=1, shuffle=False, capacity=len(run_arguments))

  # TODO currently we assume all the service_ops are the same
  service_ops, service_init_ops = service.make_graph(in_queue=in_queue,
                                                     args=args)
  service_ops = tuple(service_ops)
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
      if len(service_init_ops) > 0 and False:
          sess.run(service_init_ops)

      # its possible the service is a simple run once
      if len(service_ops) > 0:
          service_sink = pipeline.join(upstream_tensors=service_ops, capacity=8, parallel=1, multi=True)[0]
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
