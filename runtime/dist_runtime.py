import os
import tensorflow as tf
import shutil
import contextlib
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
                host, task_index = split
                try:
                    t = int(task_index)
                    if t < 0:
                        raise argparse.ArgumentTypeError("Got negative task index {t}".format(t=t))
                    parsed = host_re.match(host)
                    if parsed is None:
                        raise argparse.ArgumentTypeError("Got invalid host '{h}'. Must be HOST:PORT".format(h=host))
                    port = int(parsed.group("port"))
                    if port < 0:
                        raise argparse.ArgumentTypeError("Got negative port {p} in member '{m}'".format(p=port, m=cluster_member))
                    yield host, t
                except ValueError:
                    raise argparse.ArgumentTypeError("Unable to convert port '{p}' from member '{m}' into an integer".format(p=port_str, m=cluster_member))
        tuples = list(clusters_transformed())
        num_workers = len(tuples)
        uniq_task_indices = set(a[1] for a in tuples)
        num_uniq_workers = len(uniq_task_indices)
        if num_workers != num_uniq_workers:
            raise argparse.ArgumentTypeError("Duplicate task index specified. Got {dup} duplicate indices.\nAll: {all}\nUnique: {uniq}".format(
                dup=num_workers-num_uniq_workers,
                all=sorted(a[1] for a in tuples),
                uniq=sorted(uniq_task_indices)
            ))
        cluster = { cluster_name: dict(tuples) }
        cluster_spec = tf.train.ClusterSpec(cluster=cluster)
        return cluster_spec
    return _func

def add_default_module_args(parser):
    parser.add_argument("-T", "--task-index", type=numeric_min_checker(minimum=0, message="task index must be non-negative"), required=True, help="TF Cluster task index")
    parser.add_argument("-Q", "--queue-index", type=numeric_min_checker(minimum=0, message="queue index must be non-negative"), default=0, help="task index for cluster node that hosts the queues")
    parser.add_argument("-C", "--cluster-def", dest="cluster_spec", required=True, nargs='+', type=add_cluster_def(), help="TF Cluster definition")

@contextlib.contextmanager
def quorum(cluster_spec, task_index, session):
    def create_shutdown_queues():
        def make_shutdown_queue(idx):
            return tf.FIFOQueue(capacity=num_tasks, dtypes=tf.int32, shared_name="done_queue_{}".format(idx))

        this_idx = tf.constant(task_index, dtype=tf.int32)
        num_tasks = cluster_spec.num_tasks()
        for task_idx in cluster_spec.task_indices(cluster_name):
            with tf.device("/job:{cluster_name}/task:{idx}".format(cluster_name=cluster_name,
                                                                   idx=task_idx)):
                q = make_shutdown_queue(idx=task_idx)
                yield task_idx, (q, q.enqueue(this_idx, name="{this}_stops_{that}".format(this=task_index, that=task_idx)))
    queue_mapping = dict(create_shutdown_queues())
    all_indices = set(queue_mapping.keys())
    if task_index not in all_indices:
        raise Exception("Error on quorum setup: task index {ti} for this process not in all indices: {all}".format(
            ti=task_index, all=all_indices
        ))
    this_queue = queue_mapping[task_index][0]
    this_queue_dequeue = this_queue.dequeue(name="{cluster}_task:{t}_dequeue".format(cluster=cluster_name, t=task_index))
    stop_ops = [a[1] for idx, a in queue_mapping.items() if idx != task_index]
    needed_indices = all_indices.difference({task_index})

    def wait_for_stop():
        if len(needed_indices) == 0:
            return True
        new_idx = session.run(this_queue_dequeue)
        log.debug("Stopping. Need indices {}".format(sorted(needed_indices)))
        if new_idx in needed_indices:
            needed_indices.remove(new_idx)
        else:
            log.error("Got index {i}, even though it wasn't needed".format(i=new_idx))
        return len(needed_indices) > 0

    yield wait_for_stop # yield back to the context manager caller

    session.run(stop_ops)
    while wait_for_stop():
        pass

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
  queue_index = args.queue_index
  cluster_spec = args.cluster_spec
  for idx in (task_index, queue_index):
      # this checks if the task index is in cluster_def
      # will throw an exception if not found
      cluster_spec.task_address(job_name=cluster_name, task_index=idx)

  input_dtypes = service.input_dtypes()
  input_shapes = service.input_shapes()
  output_dtypes = service.output_dtypes()
  output_shapes = service.output_shapes()

  # TODO better define the capacity
  with tf.device("/job:{cluster_name}/task:{queue_idx}".format(cluster_name=cluster_name, queue_idx=queue_index)): # all queues live on the 0th task index
      in_queue = tf.FIFOQueue(capacity=32, dtypes=input_dtypes, shapes=input_shapes, shared_name=service+"_input")
      out_queue = tf.FIFOQueue(capacity=32, dtypes=output_dtypes, shapes=output_shapes, shared_name=service+"_output")

  with tf.device("/job:{cluster_name}/task:{task_idx}".format(cluster_name=cluster_name, task_idx=task_index)): # me
      service_ops, service_init_ops = service.make_graph(in_queue=in_queue,
                                                         args=args)
      service_ops = tuple(service_ops)
      assert len(service_ops) + len(service_init_ops) > 0

      init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer()]

      # TODO should a final join (if necessary) be moved into the service itself?
      service_sink = pipeline.join(upstream_tensors=service_ops, capacity=32, parallel=1, multi=True)[0]

      final_op = out_queue.enqueue(service_sink)
      tf.train.add_queue_runner(qr=tf.train.QueueRunner(queue=out_queue, enqueue_ops=(final_op,)))

  # start our local server
  server = tf.train.Server(cluster_spec, config=None, job_name=cluster_name, task_index=task_index)
  print("Persona distributed runtime starting TF server for index {}".format(task_index))

  with tf.Session(server.target) as sess:
      sess.run(init_ops)
      if len(service_init_ops) > 0:
          sess.run(service_init_ops)

      # its possible the service is a simple run once
      if len(service_ops) > 0:
          coord = tf.train.Coordinator()

          with quorum(cluster_spec=cluster_spec, task_index=task_index, session=sess) as wait_op:
              uninitialized_vars = tf.report_uninitialized_variables()
              while len(sess.run(uninitialized_vars)) > 0:
                  log.debug("Waiting for uninitialized variables")
                  time.sleep(startup_wait_time)

              print("All variables initialized. Persona dist executor starting {} ...".format(args.command))
              threads = tf.train.start_queue_runners(coord=coord, sess=sess)
              wait_op()
              log.debug("Got a stop from quorum. Joining...")
              coord.request_stop()
              timeout_time=60*3
              try:
                  coord.join(threads=threads, stop_grace_period_secs=timeout_time)
              except RuntimeError:
                  log.error("Unable to wait for coordinator to stop all threads after {} seconds".format(timeout_time))
              else:
                  log.debug("All threads joined and dead")
