import tensorflow as tf
from ..common import parse

class Service:
    """ A class representing a service module in Persona """

    def get_shortname(self):
        raise NotImplementedError

    def input_dtypes(self, args):
        """ Required input queue types. Default string """
        return [tf.string]

    def input_shapes(self, args):
        """ Required input queue shape. Default scalar """
        return [tf.TensorShape([])]

    def output_dtypes(self, args):
        """ Required output queue types """
        raise NotImplementedError

    def output_shapes(self, args):
        """ Required output queue shapes """
        raise NotImplementedError

    def add_graph_args(self, parser):
        """ Add the arguments to the argparse Parser (or subparser) for constructing the graph.
        These arguments typically specify the graph parameters (e.g. number of parallel nodes in graph construction) """
        raise NotImplementedError

    def add_run_args(self, parser):
        """ Add the arguments to the argparse Parser (or subparser) for RUNNING the graph
        These arguments typically specify an input file, which is the default.
        Override if you require something different """
        parse.add_dataset(parser)

    def distributed_capability(self):
        """ Whether or not this service should be exposed in a server setup.
        Override for services that do not require this capability """
        return True

    def extract_run_args(self, args):
        """ Based on what was added in `add_run_args`, extract the data from the arguments and return
         an iterable of data values to enqueue.

         The caller to this will put them into the input queue """
        raise NotImplementedError

    def make_graph(self, in_queue, args):
        """ Make the graph for this service. Returns two
        things: a list of tensors which the runtime will
        evaluate, and a list of run-once ops"""
        raise NotImplementedError

    def on_finish(self, args, results):
        """ Called by runtime when execution finished """
        pass

class ServiceSingleton:
    """ A class to wrap a service up to make a singleton instance.
    Not that those does NOT work in a multiprocessing environment. """

    _instance = None
    class_type = None # subclasses must override instance

    def __init__(self, *args, **kwargs):
        assert issubclass(self.class_type, Service)
        if self._instance is None:
            self._instance = self.class_type(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._instance, name)
