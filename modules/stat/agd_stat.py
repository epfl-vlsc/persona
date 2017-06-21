from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.python.ops import data_flow_ops, string_ops

from ..common.service import Service
from common.parse import numeric_min_checker, path_exists_checker

import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import queues, pipeline

class StatService(Service):
    """ A class representing a service module in Persona """
   
    #default inputs
    def get_shortname(self):
        return "stat"

    def output_dtypes(self):
        return []
    def output_shapes(self):
        return []
    
    def extract_run_args(self, args):
        dataset = args.dataset
        paths = [ a["path"] for a in dataset["records"] ]
        return (a for a in paths)

    def add_graph_args(self, parser):
        pass

    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two 
        things: a list of tensors which the runtime will 
        evaluate, and a list of run-once ops"""
        # make the graph
        dataset = args.dataset
        total_reads = 0
        chunk_size = dataset['records'][0]['last'] - dataset['records'][0]['first']
        for chunk in dataset['records']:
            total_reads += chunk['last'] - chunk['first']
        print("Total Reads: {}".format(total_reads))
        print("AGD Chunk size: {}".format(chunk_size))
        print("Columns present: {}".format(dataset['columns']))
        if 'sort' in dataset:
            print("Sort order: {}".format(dataset['sort']))
        else:
            print("Sort order unspecified")
        if 'results' in dataset['columns']:
          print("Aligned to reference: {}".format(dataset['reference']))

        run_once = [tf.constant(0)]
        return [], run_once 

