import os
import tensorflow as tf
import shutil
from tensorflow.contrib.persona import pipeline
from common import parse
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

def gen_unique_filename(prefix, suffix, start=0):
    yield prefix+suffix
    while True:
        yield "{p}_{num}{suffix}".format(p=prefix, num=start, suffix=suffix)
        start+=1

def create_unique_file(directory, prefix, suffix, start=0):
    if not os.path.isdir(directory):
        raise Exception("Unable to create unique file in directory {}".format(directory))
    flname_gen = (os.path.join(directory, a) for a in gen_unique_filename(prefix=prefix, suffix=suffix, start=start))
    for filename in flname_gen:
        if not os.path.exists(filename):
            return filename

def add_default_module_args(parser):
    cwd = os.getcwd()
    parser.add_argument("--record", default=False, action='store_true', help="record usage of the running process")
    parser.add_argument("--record-directory", default=cwd, type=parse.path_exists_checker(), help="directory to store runtime statistics")
    parser.add_argument("--summary", default=False, action="store_true", help="Add TensorFlow summary info to the graph")
    parser.add_argument("--summary-directory", default=os.path.join(cwd, "traces"), type=parse.path_exists_checker(make_if_empty=True), help="directory to record summary information into")

def create_variables(variable_inits):
    for name, value in variable_inits.items():
        yield tf.get_variable(name=name, initializer=value, trainable=False)

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

  in_queue = tf.train.input_producer(input_tensor=run_arguments, num_epochs=1, shuffle=False, capacity=len(run_arguments))

  # TODO currently we assume all the service_ops are the same
  service_ops, service_init_ops = service.make_graph(in_queue=in_queue,
                                                     args=args)
  if not isinstance(service_ops, list):
      service_ops = list(service_ops)
  assert len(service_ops) + len(service_init_ops) > 0

  has_service_ops = len(service_ops) > 0
  if has_service_ops:
      service_sink = pipeline.join(upstream_tensors=service_ops, capacity=64, parallel=1, multi=True, name="global_sink_queue")

  variables = tuple(create_variables(variable_inits=service.variables))

  init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer()]

  # service graph may have summary nodes
  summary = args.summary if hasattr(args, 'summary') else False

  results = []
  stats_results = {}
  with tf.Session() as sess:
      if summary and has_service_ops:
          trace_dir = setup_output_dir(dirname=args.command + "_summary")
          service_sink.append(tf.summary.merge_all())
          summary_writer = tf.summary.FileWriter(trace_dir, graph=sess.graph, max_queue=2**20, flush_secs=10**4)
      else:
          summary = False

      count = 0
      sess.run(init_ops)
      if len(service_init_ops) > 0:
          res = sess.run(service_init_ops)
          results.append(res)
          if summary:
              results.append(res[:-1])
          else:
              results.append(res)

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

          variable_values = [ sess.run(v) for v in variables ]
          service.on_finish(args, results, variables=variable_values)
  if summary:
      summary_writer.flush(); summary_writer.close()
  if record_stats:
      params = vars(args)
      del params["func"]
      stats_results["params"] = vars(args)
      with open(create_unique_file(directory=stats_directory, prefix="runtime_stats", suffix=".json"), 'w+') as fl:
        json.dump(stats_results, fl)
