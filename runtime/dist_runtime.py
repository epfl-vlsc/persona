import os
import tensorflow as tf
import shutil
import argparse
import re
import time
from tensorflow.contrib.persona import pipeline
from modules.common.parse import numeric_min_checker

import logging
logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

cluster_name = "persona"
startup_wait_time = 1
host_re = re.compile("(?P<host>(?:\w+)(?:\.\w+)*):(?P<port>\d+)")

def add_cluster_def():
    def _func(cluster_members):
        def clusters_transformed():
            for cluster_member in cluster_members:
                split = cluster_member.split(",")
                if len(split) != 2:
                    raise argparse.ArgumentTypeError("Got badly formed cluster member: '{}'".format(cluster_member))
                host, port_str = split
                try:
                    port = int(port_str)
                    if host_re.match(host) is None:
                        raise argparse.ArgumentTypeError("Got invalid host '{h}'. Must be HOST:PORT".format(h=host))
                    if port < 0:
                        raise argparse.ArgumentTypeError("Got negative port {p} in member '{m}'".format(p=port, m=cluster_member))
                    yield host, port
                except ValueError:
                    raise argparse.ArgumentTypeError("Unable to convert port '{p}' from member '{m}' into an integer".format(p=port_str, m=cluster_member))
        cluster = { cluster_name: list(clusters_transformed()) }
        cluster_spec = tf.train.ClusterSpec(cluster=cluster)
        return cluster_spec
    return _func

def add_default_module_args(parser):
    parser.add_argument("-T", "--task-index", type=numeric_min_checker(minimum=0, message="task index must be non-negative"), required=True, help="TF Cluster task index")
    parser.add_argument("-C", "--cluster-def", required=True, nargs='+', type=add_cluster_def(), help="TF Cluster definition")

def setup_output_dir(dirname="cluster_traces"):
    trace_path = os.path.join(os.path.dirname(os.getcwd()), dirname)
    if os.path.exists(trace_path):
        # nuke it
        shutil.rmtree(trace_path)
    os.makedirs(trace_path)
    return trace_path

def execute(args, modules):
 
  module = modules[args.command]

  if hasattr(args, 'service'):
    service_mode = args.service
    service = module.lookup_service(name=service_mode)
  else:
    # there is only one service if the args does not have .service
    service = module.get_services()[0]

  if not service.distributed_capability():
    raise Exception("Service {} does not support distributed execution".format(args.service))

  task_index = args.task_index
  input_dtypes = service.input_dtypes()
  input_shapes = service.input_shapes()
  output_dtypes = service.output_dtypes()
  output_shapes = service.output_shapes()

  # we assume that the input queue has name `service`_input and is hosted in the cluster
  # TODO better define the capacity ?
  in_queue = tf.FIFOQueue(capacity=32, dtypes=input_dtypes, shapes=input_shapes, shared_name=service+"_input")
  # assuming for now that input is the same as output shape and type (generally, string keys for AGD chunks)
  out_queue = tf.FIFOQueue(capacity=32, dtypes=output_dtypes, shapes=output_shapes, shared_name=service+"_output")

  with tf.device("/job:worker/task:"+task_index): # me

      service_ops, service_init_ops = service.make_graph(in_queue=in_queue,
                                                         args=args)
      service_ops = tuple(service_ops)
      assert len(service_ops) + len(service_init_ops) > 0

      init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer()]

      # TODO should a final join (if necessary) be moved into the service itself?
      service_sink = pipeline.join(upstream_tensors=service_ops, capacity=8, parallel=1, multi=True)[0]

      final_op = out_queue.enqueue(service_sink)
      # service graph may have summary nodes
      # TODO figure out distributed summaries
      merged = tf.summary.merge_all()
      summary = args.summary if hasattr(args, 'summary') else False

  # cluster def should be hostname:port
  workers = [ a for a in args.cluster_def ]
  cluster = { "worker" : workers }
  clusterspec = tf.train.ClusterSpec(cluster).as_cluster_def()

  # start our local server
  server = tf.train.Server(clusterspec, config=None, job_name="worker", task_index=task_index)
  print("Persona distributed runtime starting TF server for index {}".format(task_index))

  results = []
  # run our local graph 
  with tf.Session(server.target) as sess:
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
          uninitialized_vars = tf.report_uninitialized_variables()
          while len(sess.run(uninitialized_vars)) > 0:
              log.debug("Waiting for uninitialized variables")
              time.sleep(1)
          coord = tf.train.Coordinator()
          print("Persona dist executor starting {} ...".format(args.command))
          threads = tf.train.start_queue_runners(coord=coord, sess=sess)
          while not coord.should_stop():
              try:
                  print("Persona dist running round {}".format(count))
                  result = sess.run(final_op)
                  count += 1
                  if summary:
                      summary_writer.add_summary(result[-1], global_step=count)
              except tf.errors.OutOfRangeError:
                  print('Got out of range error!')
                  break
          print("Coord requesting stop")
          coord.request_stop()
          coord.join(threads, stop_grace_period_secs=10)

