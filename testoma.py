
import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import queues, pipeline

import numpy as np
#cimport numpy as np
import json
import sys
import os
from six import string_types
import re
import math
import ipdb

def data_dir():
    default_data_dir = os.path.join(test_dir(), 'data/')

    if not os.path.exists(default_data_dir):
        raise IOError('The default data directory does not exists at %s!' % default_data_dir)

    return default_data_dir


def matrix_dir():
    default_matrix_dir = os.path.join('.', 'data/matrices/json/')

    if not os.path.exists(default_matrix_dir):
        raise IOError('The default matrix directory does not exists at %s!' % default_matrix_dir)

    return default_matrix_dir



def load_default_environments():
  default_envs = {
      'environments': read_all_env_json(os.path.join(matrix_dir(), 'all_matrices.json')),
      'log_pam1': read_env_json((os.path.join(matrix_dir(), 'logPAM1.json')))
      }
  return default_envs

def scale_to_byte(val, factor):
    """
    Scales the given double value to a byte value. This is needed to create the matrix and the gapOpen/Ext costs
    from the double versions, for the byte alignment function

    :param val: the double value which we want to scale up
    :param factor: the factor used for the scaling
    :return: a scaled integer from the -128..127 interval
    """
    ret = val * factor

    #for an upper bound estimation overflow is a problem, but underflow is not
    if ret > 127:
        raise OverflowError("Scaling overflow in scale_to_byte factor = %f, doubleValue = %f, result = %f" %
                (factor, val, ret))

    if ret < -128:
        return -128

    return int(math.ceil(ret))


def scale_to_short(val, factor):
    """
    Scales the given double value to a short value. This is needed to create the matrix and the gapOpen/Ext costs
    from the double versions, for the short alignment function

    :param val: the double value which we want to scale up
    :param factor: the factor used for the scaling
    :return: a scaled integer from the -32768..32767 interval
    """
    ret = val * factor

    #for an upper bound estimation overflow is a problem, but underflow is not
    if ret > 32767:
        raise OverflowError("Scaling overflow in scaleToShort factor = %f, doubleValue = %f, result = %f" %
                        (factor, val, ret))

    if ret < -32768:
        return -32768


    return int(math.ceil(ret))
    
def create_scaled_matrices(matrix, gap_open, gap_ext, threshold):
    """
    Creates the int8 and int16 matrix and gap costs from the double matrix by using a simple scaling
    :return:nothing
    """

    factor = byte_factor(matrix, threshold)
    int8_matrix = np.vectorize(lambda x: scale_to_byte(x, factor))(matrix ).astype(np.int8)
    int8_gap_open = scale_to_byte(gap_open, factor)
    int8_gap_ext = scale_to_byte(gap_ext, factor)

    factor = short_factor(threshold)
    int16_matrix = np.vectorize(lambda x: scale_to_short(x, factor))(matrix ).astype(np.int16)
    int16_gap_open = scale_to_short(gap_open, factor)
    int16_gap_ext = scale_to_short(gap_ext, factor)

    return (int8_matrix, int8_gap_open, int8_gap_ext, int16_matrix, int16_gap_open, int16_gap_ext)

#calculates the byte scaling factor
def byte_factor(matrix, threshold):

    #This is copied from the C code and I have no idea why we use this at the byte version but not at the short
    abs_min = abs(np.amin(matrix))  # the float64 matrix 
    return 255.0/(threshold + abs_min)


def short_factor(threshold):
    return 65535.0 / threshold

def create_environment(gap_open, gap_ext, pam_distance, scores, column_order, threshold=85.0, **kwargs):
    """
    Creates an environment from the given parameters.
    :param gap_open: gap opening cost
    :param gap_ext: gap extension cost
    :param pam_distance: pam distance
    :param scores: distance matrix
    :param column_order: column order of the distance matrix
    :param kwargs:
    :return: an AlignmentEnvironment
    """

    reg = re.compile('^[A-Z]*$')
    column_order = ''.join(column_order)

    if not reg.match(column_order):
        raise ValueError("Could not create environment with columns '%s', because it contains invalid characters." %
                         column_order)

    if len(scores) != len(column_order):
        raise ValueError('The dimension of the matrix is not consistent with the column order')

    #TODO check whether gap_open <= gap_ext?


    threshold = threshold
    gap_open = gap_open
    gap_ext = gap_ext
    pam = pam_distance
    compact_matrix = scores

    #convert the compact matrix into C compatible one by extending it to a 26x26 one
    extended_matrix = [[0 for x in range(26)] for x in range(26)]
    for i in range(0, len(column_order)):
        if len(compact_matrix[i]) != len(column_order):
            raise ValueError('The dimension of the matrix is not consistent with the column order')
        for j in range(0, len(column_order)):
            extended_matrix[ord(column_order[i]) - ord('A')][ord(column_order[j]) - ord('A')] = compact_matrix[i][j]

    float64_matrix = np.array(extended_matrix, dtype=np.float64)

    scaled = create_scaled_matrices(float64_matrix, gap_open, gap_ext, threshold)

    return (threshold, gap_open, gap_ext, pam_distance, float64_matrix) + scaled


def read_env_json(json_data):
    """
    This function reads an AlignmentEnvironment from a JSON object or a file that contains the JSON data
    :param json_data: the JSON object from which we want to read the environment or a JSON file
    :return: the environment
    """
    if isinstance(json_data, string_types):
        with open(json_data) as f:
            json_data = json.load(f)

    return create_environment(**json_data)


def read_all_env_json(file_loc):
    """
    Reads all of the alignment environments from the given json file.

    :param file_loc: the location where the matrices are stored
    :return: a list of AlignmentEnvironments
    """
    with open(file_loc) as json_file:
        json_data = json.load(json_file)

    ret = []

    for matrix_json in json_data['matrices']:
        ret.append(read_env_json(matrix_json))

    return ret

float_matrices = []
int8_matrices = []
int16_matrices = []
gaps = []
gap_extends = []
int8_gaps = []
int8_gap_extends = []
int16_gaps = []
int16_gap_extends = []
pam_dists = []
thresholds = []

file_loc = os.path.join(matrix_dir(), 'all_matrices.json')
with open(file_loc) as json_file:
    json_data = json.load(json_file)

p = True
for matrix_json in json_data['matrices']:
    threshold, gap_open, gap_ext, pam_distance, float64_matrix, int8_matrix, int8_gap_open, int8_gap_ext, int16_matrix, int16_gap_open, int16_gap_ext = read_env_json(matrix_json)
    thresholds.append(threshold)
    gaps.append(gap_open)
    gap_extends.append(gap_ext)
    pam_dists.append(pam_distance)
    float_matrices.append(tf.make_tensor_proto(float64_matrix))

logpam_loc = os.path.join(matrix_dir(), 'logPAM1.json')
with open(logpam_loc) as json_file:
    logpam_json_data = json.load(json_file)

lp_thresh, lp_gap, lp_gap_ext, lp_pam, lp_matrix, lpint8_matrix, lpint8_gap_open, lpint8_gap_ext, lpint16_matrix, lpint16_gap_open, lpint16_gap_ext = read_env_json(logpam_json_data)

print("there are {} entries".format(len(float_matrices)))
op = persona_ops.alignment_environments(gaps=tf.make_tensor_proto(gaps, dtype=tf.float64), gap_extends=tf.make_tensor_proto(gap_extends, dtype=tf.float64), pam_dists=tf.make_tensor_proto(pam_dists, dtype=tf.float64), 
                                        thresholds=tf.make_tensor_proto(thresholds, dtype=tf.float64), double_matrices=float_matrices, logpam_matrix=tf.make_tensor_proto(lp_matrix), 
                                        logpam_gap=tf.make_tensor_proto(lp_gap, dtype=tf.float64), logpam_gap_ext=tf.make_tensor_proto(lp_gap_ext, dtype=tf.float64),
                                        logpam_pam_dist=tf.make_tensor_proto(lp_pam, dtype=tf.float64), logpam_threshold=tf.make_tensor_proto(lp_thresh, dtype=tf.float64))

print(os.getpid())
ipdb.set_trace()

print("about to run op!")
init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer()]
with tf.Session() as sess:
  sess.run(init_ops)
  res = sess.run(op)

