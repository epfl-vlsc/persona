#!/usr/bin/env python3
from __future__ import print_function


class Service:
    """ A class representing a service module in Persona """
    
    def input_dtypes(self):
        """ Required input queue types. Default string """
        return [tf.string]

    def input_shapes(self):
        """ Required input queue shape. Default scalar """
        return [tf.TensorShape([1])]

    def output_dtypes(self):
        """ Required output queue types """
        raise NotImplementedError

    def output_shapes(self):
        """ Required output queue shapes """
        raise NotImplementedError

    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two 
        things: a list of tensors which the runtime will 
        evaluate, and a list of run-once ops"""
        raise NotImplementedError

    def on_finish(self, args):
        """ Called by runtime when execution finished """
        raise NotImplementedError
