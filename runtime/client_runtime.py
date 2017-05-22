import os
import tensorflow as tf
import shutil
from . import dist_common
from common import parse
from tensorflow.contrib.persona import pipeline

import logging
logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

def add_default_module_args(parser):
    parser.add_argument("-Q", "--queue-index", type=parse.numeric_min_checker(minimum=0, message="queue index must be non-negative"), default=0, help="task index for cluster node that hosts the queues")
    # TODO we want to have sensible defaults for this eventually!
    parser.add_argument("--queue-host", required=True, help="host running the queue service")
    parser.add_argument("--queue-port", type=parse.numeric_min_checker(0, "port must be >0"), required=True, help="port of the host running the queue service")

def execute(args, modules):
  queue_index = args.queue_index

  module = modules[args.command]

  if hasattr(args, 'service'):
    service_mode = args.service
    service = module.lookup_service(name=service_mode)
  else:
    # there is only one service if the args does not have .service
    service = module.get_services()[0]

  if not service.distributed_capability():
    raise Exception("Service {} does not support distributed execution".format(args.service))

  run_arguments = tuple(service.extract_run_args(args=args))
  input_dtypes = service.input_dtypes()
  input_shapes = service.input_shapes()
  output_dtypes = service.output_dtypes()
  output_shapes = service.output_shapes()

  results = []
  # run our local graph
  target
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

