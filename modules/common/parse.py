from argparse import ArgumentTypeError
import json
import os

import logging
logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

filepath_key = "filepath"
def add_dataset(parser):
    """
    Adds the dataset, including parsing, to any parser / subparser
    """
    def dataset_parser(filename):
        if not os.path.isfile(filename):
            raise ArgumentTypeError("AGD metadata file not present at {}".format(filename))
        with open(filename) as f:
            try:
                loaded = json.load(f)
                loaded[filepath_key] = filename
                return loaded
            except json.JSONDecodeError:
                log.error("Unable to parse AGD metadata file {}".format(filename))
                raise

    parser.add_argument("dataset", type=dataset_parser, help="The AGD json metadata file describing the dataset")
