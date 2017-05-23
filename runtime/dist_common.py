import re
import argparse
import contextlib
from common import parse
import tensorflow as tf

import logging
logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

cluster_name = "persona"
host_re = re.compile("(?P<host>(?:\w+)(?:\.\w+)*):(?P<port>\d+)")

def parse_cluster_def_member(cluster_member):
    split = cluster_member.split(",")
    if len(split) != 2:
        raise argparse.ArgumentTypeError("Got badly formed cluster member: '{}'".format(cluster_member))
    task_index, host = split
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
        return t, host
    except ValueError:
        raise argparse.ArgumentTypeError("Unable to convert task index {ti} into an integer".format(ti=task_index))

def queue_only_args(parser):
    parser.add_argument("-Q", "--queue-index", type=parse.numeric_min_checker(minimum=0, message="queue index must be non-negative"), default=0, help="task index for cluster node that hosts the queues")
    parser.add_argument("-C", "--cluster", dest="cluster_members", required=True, nargs='+', type=parse_cluster_def_member, help="TF Cluster definition")

def make_cluster_spec(cluster_members):
    num_members = len(cluster_members)
    unique_indices = { a[0] for a in cluster_members }
    num_uniq_indices = len(unique_indices)
    if num_uniq_indices < num_members:
        raise Exception("Not all indices for cluster spec are uniq. {} items are duplicates".format(num_members - num_uniq_indices))
    cluster = { cluster_name: dict(cluster_members) }
    cluster_spec = tf.train.ClusterSpec(cluster=cluster)
    return cluster_spec

@contextlib.contextmanager
def quorum(cluster_spec, task_index, session):
    def create_shutdown_queues():
        def make_shutdown_queue(idx):
            return tf.FIFOQueue(capacity=num_tasks, dtypes=tf.int32, shared_name="done_queue_{}".format(idx))

        this_idx = tf.constant(task_index, dtype=tf.int32)
        num_tasks = cluster_spec.num_tasks(job_name=cluster_name)
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
    this_queue_dequeue = this_queue.dequeue(name="{cluster}_task_{t}_dequeue".format(cluster=cluster_name, t=task_index))
    stop_ops = [a[1] for idx, a in queue_mapping.items() if idx != task_index]
    needed_indices = all_indices.difference({task_index})

    def wait_for_stop():
        if len(needed_indices) == 0:
            return False
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

def make_queue_device_name(cluster_name, queue_index):
    return "/job:{cluster_name}/task:{queue_idx}".format(cluster_name=cluster_name, queue_idx=queue_index)

def make_common_queues(service_name, queue_index, cluster_name, input_dtypes, input_shapes, output_dtypes, output_shapes, in_capacity=4096, out_capacity=4096):
    log.debug("Making queues for service_name: {}".format(service_name))
    queue_device = make_queue_device_name(cluster_name=cluster_name, queue_index=queue_index)
    input_name = service_name+"_input"
    output_name = service_name+"_output"
    with tf.device(queue_device):
        in_queue = tf.FIFOQueue(capacity=in_capacity, dtypes=input_dtypes, shapes=input_shapes, shared_name=input_name, name=input_name)
        out_queue = tf.FIFOQueue(capacity=out_capacity, dtypes=output_dtypes, shapes=output_shapes, shared_name=output_name, name=output_name)
    return in_queue, out_queue
