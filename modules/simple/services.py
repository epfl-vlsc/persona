from ..common.service import Service
import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import queues, pipeline

class PrintFinish(Service):
    def on_finish(self, args, results, variables):
        print("Got results: {}".format(results))

class EchoService(PrintFinish):
    def get_shortname(self):
        return "echo"

    def extract_run_args(self, args):
        return args.strings

    def output_dtypes(self, *args, **kwargs):
        return self.input_dtypes(*args, **kwargs)

    def output_shapes(self, *args, **kwargs):
        return self.input_shapes(*args, **kwargs)

    def add_run_args(self, parser):
        parser.add_argument("strings", nargs="+", help="one or more strings to echo through the system")

    def add_graph_args(self, parser):
        pass

    def make_graph(self, in_queue, args):
        return ((in_queue.dequeue(),),), []

class Incrementer(PrintFinish):
    def get_shortname(self):
        return "increment"

    def extract_run_args(self, args):
        return args.integers

    def input_dtypes(self, *args, **kwargs):
        return (tf.int64,)

    def output_shapes(self, *args, **kwargs):
        return self.input_shapes(*args, **kwargs)

    def output_dtypes(self, *args, **kwargs):
        return self.input_dtypes(*args, **kwargs)

    def add_run_args(self, parser):
        parser.add_argument("integers", type=int, nargs="+", help="one or more integers to increment")

    def add_graph_args(self, parser):
        parser.add_argument("-i", "--increment", type=int, default=1, help="amount to increment values by")

    def make_graph(self, in_queue, args):
        increment = args.increment
        incr_by = tf.constant(increment, dtype=tf.int64)
        incr_op = tf.to_int64(in_queue.dequeue()) + incr_by
        ready_to_process = pipeline.join(upstream_tensors=(incr_op,),
                                         parallel=1,
                                         capacity=1,
                                         multi=False, name="ready_to_process")
        return (ready_to_process,), []
