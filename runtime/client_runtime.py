import os
import tensorflow as tf
import shutil
from . import dist_common
from .dist_common import cluster_name
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

  module = modules[args.client_command]

  if hasattr(args, 'service'):
    service_mode = args.service
    service = module.lookup_service(name=service_mode)
  else:
    # there is only one service if the args does not have .service
    service = module.get_services()[0]

  if not service.distributed_capability():
    raise Exception("Service {} does not support distributed execution".format(args.service))

  run_arguments = tuple(service.extract_run_args(args=args))
  input_dtypes = service.input_dtypes(args=args)
  input_shapes = service.input_shapes(args=args)
  output_dtypes = service.output_dtypes(args=args)
  output_shapes = service.output_shapes(args=args)
  service_name = service.get_shortname()

  with tf.device("/job:{cluster_name}/task:{queue_idx}".format(cluster_name=cluster_name, queue_idx=queue_index)): # all queues live on the 0th task index
      in_queue = tf.FIFOQueue(capacity=32, dtypes=input_dtypes, shapes=input_shapes, shared_name=service_name+"_input")
      out_queue = tf.FIFOQueue(capacity=32, dtypes=output_dtypes, shapes=output_shapes, shared_name=service_name+"_output")

  enqueue_op = in_queue.enqueue_many(vals=(run_arguments,), name=service_name+"_client_enqueue")
  dequeue_op = out_queue.dequeue_many(n=len(run_arguments))

  # run our local graph
  queue_host = args.queue_host
  queue_port = args.queue_port
  target = "grpc://{host}:{port}".format(host=queue_host, port=queue_port)
  with tf.Session(target=target) as sess:
      sess.run(enqueue_op)
      try:
          results = sess.run(dequeue_op)
          service.on_finish(args=args, results=results)
      except tf.errors.OutOfRangeError:
          log.error("Got out of range error! Session: {}".format(target))
