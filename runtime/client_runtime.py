import os
import tensorflow as tf
import shutil
import contextlib
import threading
from . import dist_common
import time
from .dist_common import cluster_name
from common import parse
from tensorflow.contrib.persona import pipeline

import logging
logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

startup_wait_time = 1

@contextlib.contextmanager
def enqueue_items(sess, enqueue_all_op, timeout=10):
    def enqueue_func():
        sess.run(enqueue_all_op)
    enqueue_thread = threading.Thread(target=enqueue_func, name="enqueue_function")
    enqueue_thread.start()
    yield
    log.debug("Joining enqueue thread")
    enqueue_thread.join()
    log.debug("Enqueue thread is dead")

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
  service_name = args.client_command + "_" + service.get_shortname()

  in_queue, out_queue = dist_common.make_common_queues(service_name=service_name,
                                                       queue_index=queue_index,
                                                       cluster_name=cluster_name,
                                                       input_dtypes=input_dtypes,
                                                       input_shapes=input_shapes,
                                                       output_dtypes=output_dtypes,
                                                       output_shapes=output_shapes)

  uniq_lens = set(len(a) for a in run_arguments)
  if len(uniq_lens) != 1:
      raise Exception("all run arguments must be the same length. Got lengths {}".format(uniq_lens))
  transposed = [list(i)for i in zip(*run_arguments)]
  enqueue_op = in_queue.enqueue_many(vals=transposed, name=service_name+"_client_enqueue")
  dequeue_single_op = out_queue.dequeue(name="client_dequeue")
  expected_result_count = len(run_arguments) # FIXME we're just making this assumption for now!

  # run our local graph
  queue_host = args.queue_host
  queue_port = args.queue_port
  target = "grpc://{host}:{port}".format(host=queue_host, port=queue_port)
  results = []
  with tf.Session(target=target) as sess:
      uninitialized_vars = tf.report_uninitialized_variables()
      while len(sess.run(uninitialized_vars)) > 0:
          log.debug("Waiting for uninitialized variables")
          time.sleep(startup_wait_time)
      log.debug("All variables initialized. Persona dist executor starting {} ...".format(args.command))
      with enqueue_items(sess=sess, enqueue_all_op=enqueue_op):
          try:
              while expected_result_count > 0:
                  next_result = sess.run(dequeue_single_op)
                  # TODO get rid of this!
                  log.debug("Got result: {}".format(next_result))
                  expected_result_count -= 1
                  results.append(next_result)
              service.on_finish(args=args, results=results)
          except tf.errors.OutOfRangeError:
              log.error("Got out of range error! Session: {}".format(target))
