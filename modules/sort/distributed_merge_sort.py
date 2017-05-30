import os
import json
from ..common.service import Service
from ..common import bridge
from common.parse import numeric_min_checker, path_exists_checker, non_empty_string_checker
from tensorflow.python.ops import io_ops, variables, string_ops, array_ops, data_flow_ops, math_ops, control_flow_ops
from tensorflow.python.framework import ops, tensor_shape, common_shapes, constant_op, dtypes

import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import queues, pipeline

def add_common_args(parser):
    parser.add_argument("-b", "--order-by", default="location", choices=["location", "metadata"], help="sort by this parameter [location | metadata]")
    parser.add_argument("-w", "--write-parallel", default=1, help="write pipeline parallelism",
                        type=numeric_min_checker(minimum=1, message="writing pipeline min"))

class SortCommonService(Service):
    def add_graph_args(self, parser):
        parser.add_argument("-r", "--read-parallel", default=1, type=numeric_min_checker(minimum=1, message="read parallelism min for sort phase"),
                            help="read parallelism")
        parser.add_argument("-p", "--process-parallel", default=1, type=numeric_min_checker(minimum=1, message="process parallelism min for sort phase"),
                            help="chunk processing parallelism level")
        parser.add_argument("-s", "--sort-parallel", default=1, help="parallel sorting pipelines",
                            type=numeric_min_checker(minimum=1, message="sorting pipeline min"))
        parser.add_argument("-c", "--column-grouping", default=5, help="grouping factor for chunks",
                            type=numeric_min_checker(minimum=1, message="column grouping min"))
        add_common_args(parser=parser)

    def input_dtypes(self, args):
        """ Default is (record_id, file_path) """
        return [tf.string] * 2

    def input_shapes(self, args):
        """ Default is (record_id, file_path) """
        return [tf.TensorShape([])] * 2

    def output_dtypes(self, args):
        """ Default is (record_id, intermediate_file_path) """
        return [tf.string] * 2

    def output_shapes(self, args):
        """ Default is (record_id, intermediate_file_path) """
        return [tf.TensorShape([])] * 2

class MergeCommonService(Service):
    def add_graph_args(self, parser):
        parser.add_argument("-c", "--chunk", default=100000, type=numeric_min_checker(1, "need non-negative chunk size"),
                            help="final merge output chunk size")
        add_common_args(parser=parser)

class CephSort(SortCommonService):
    def get_shortname(self):
        return "ceph_distributed_sort"

class CephMerge(MergeCommonService):
    def get_shortname(self):
        return "ceph_distributed_merge"
