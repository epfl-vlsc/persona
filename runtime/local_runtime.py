import os
import tensorflow as tf
import shutil
from tensorflow.contrib.persona import pipeline
from modules.common import parse
from common import recorder
import contextlib
import json

def setup_output_dir(dirname="cluster_traces"):
    trace_path = os.path.join(os.path.dirname(os.getcwd()), dirname)
    if os.path.exists(trace_path):
        # nuke it
        shutil.rmtree(trace_path)
    os.makedirs(trace_path)
    return trace_path

def add_default_module_args(parser):
    parser.add_argument("--record", default=False, action='store_true', help="record usage of the running process")
    parser.add_argument("--record-directory", default=os.path.dirname(os.getcwd()), type=parse.path_exists_checker(), help="directory to store runtime statistics")

def execute(args, modules):
  record_stats = args.record
  stats_directory = args.record_directory
 
  module = modules[args.command]

  if hasattr(args, 'service'):
    service_mode = args.service
    service = module.lookup_service(name=service_mode)
  else:
    # there is only one service if the args does not have .service
    service = module.get_services()[0]
    
  run_arguments = tuple(service.extract_run_args(args=args))
  input_dtypes = service.input_dtypes()
  input_shapes = service.input_shapes()

  in_queue = tf.train.input_producer(input_tensor=run_arguments, num_epochs=1, shuffle=False, capacity=len(run_arguments))

  # TODO currently we assume all the service_ops are the same
  service_ops, service_init_ops = service.make_graph(in_queue=in_queue,
                                                     args=args)
  if not isinstance(service_ops, list):
      service_ops = list(service_ops)
  assert len(service_ops) + len(service_init_ops) > 0

  service_sink = pipeline.join(upstream_tensors=service_ops, capacity=64, parallel=1, multi=True, name="global_sink_queue")[0]

  init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer()]

  # service graph may have summary nodes
  summary = args.summary if hasattr(args, 'summary') else False

  results = []
  stats_results = {}
  with tf.Session() as sess:
      if summary:
          trace_dir = setup_output_dir(dirname=args.command + "_summary")
          service_sink.append(tf.summary.merge_all())
          summary_writer = tf.summary.FileWriter(trace_dir, graph=sess.graph, max_queue=2**20, flush_secs=10**4)

      count = 0
      sess.run(init_ops)
      if len(service_init_ops) > 0:
          sess.run(service_init_ops)

      # its possible the service is a simple run once
      if len(service_ops) > 0:
          with contextlib.ExitStack() as stack:
              if record_stats:
                  stack.enter_context(recorder.UsageRecorder(stats_results))
              coord = tf.train.Coordinator()
              print("Local executor starting {} ...".format(args.command))
              threads = tf.train.start_queue_runners(coord=coord, sess=sess)
              while not coord.should_stop():
                  try:
                      #print("Running round {}".format(count))
                      result = sess.run(service_sink)
                      count += 1
                      if summary:
                          results.append(result[:-1])
                          summary_writer.add_summary(result[-1], global_step=count)
                      else:
                          results.append(result)
                  except tf.errors.OutOfRangeError:
                      #print('Got out of range error!')
                      break
              print("Local executor finishing ...")
              coord.request_stop()
              coord.join(threads, stop_grace_period_secs=10)

          service.on_finish(args, results)
  if record_stats:
      with open(os.path.join(stats_directory, "runtime_stats.json"), 'w+') as fl:
        json.dump(stats_results, fl)