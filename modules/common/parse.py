import argparse
import json
import os

import logging
logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

def numeric_min_checker(minimum, message, numeric_type=int):
    def check_number(n):
        n = numeric_type(n)
        if n < minimum:
            raise argparse.ArgumentTypeError("{msg}: got {got}, minimum is {minimum}".format(
                msg=message, got=n, minimum=minimum
            ))
        return n
    return check_number

def add_dataset(parser):
    """
    Adds the dataset, including parsing, to any parser / subparser
    """
    def dataset_parser(filename):
        if not os.path.isfile(filename):
            raise argparse.ArgumentTypeError("AGD metadata file not present at {}".format(filename))
        with open(filename) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                log.error("Unable to parse AGD metadata file {}".format(filename))
                raise

    parent_parser.add_argument("dataset", type=dataset_parser, help="The AGD json metadata file describing the dataset")
