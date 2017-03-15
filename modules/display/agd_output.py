#!/usr/bin/env python3
import tensorflow as tf
import argparse
import os
import json

persona_ops = tf.contrib.persona.persona_ops()

def get_args():
    parser = argparse.ArgumentParser(description="An output utility for AGD Format",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("json_file", help="The json file describing the format")
    parser.add_argument("start", type=int, help="The absolute index at which to start printing records")
    parser.add_argument("finish", type=int, help="The absolute index at which to stop printing records")
    parser.add_argument("-u", "--unpack", default=True, action='store_false', help="Whether or not to unpack binary bases")
    
    args = parser.parse_args()

    if not os.path.isabs(args.json_file):
        args.json_file = os.path.abspath(args.json_file)

    return args

def run(args):

  with open(args.json_file, 'r') as j:
    dataset_params = json.load(j)

  records = dataset_params['records']
  first_record = records[0]
  chunk_size = first_record["last"] - first_record["first"]
  chunknames = []
  for record in records:
    chunknames.append(record['path'])

  if (args.finish <= args.start):
    args.finish = args.start + 1


  pathname = os.path.dirname(args.json_file) + '/'

  path = tf.constant(pathname)
  start = tf.constant(args.start, dtype=tf.int32)
  finish = tf.constant(args.finish, dtype=tf.int32)
  names = tf.constant(chunknames)
  size = tf.constant(chunk_size)
  unpack = tf.constant(args.unpack)
  output = persona_ops.agd_output(path, names, size, start, finish)
    
  init_op = tf.initialize_all_variables()

  #print(os.getpid())
  #import ipdb; ipdb.set_trace()
  with tf.Session() as sess:
    sess.run([init_op])
    sess.run([output])

if __name__ == "__main__":
    run(get_args())
