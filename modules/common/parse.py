from argparse import ArgumentTypeError
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
            raise ArgumentTypeError("{msg}: got {got}, minimum is {minimum}".format(
                msg=message, got=n, minimum=minimum
            ))
        return n
    return check_number

def path_exists_checker(check_dir=True, make_absolute=True, make_if_empty=False):
    def _func(path):
        path = os.path.expanduser(path)
        if os.path.exists(path):
            if check_dir:
                if not os.path.isdir(path):
                    raise ArgumentTypeError("path {pth} exists, but isn't a directory".format(pth=path))
            elif not os.path.isfile(path=path):
                raise ArgumentTypeError("path {pth} exists, but isn't a file".format(pth=path))
        elif check_dir and make_if_empty:
            os.makedirs(name=path)
        else:
            raise ArgumentTypeError("path {pth} doesn't exist on filesystem".format(pth=path))
        if make_absolute:
            path = os.path.abspath(path=path)
        return path
    return _func

def non_empty_string_checker(string):
    if len(string) == 0:
        raise ArgumentTypeError("string is empty!")
    return string

def add_dataset(parser):
    """
    Adds the dataset, including parsing, to any parser / subparser
    """
    def dataset_parser(filename):
        if not os.path.isfile(filename):
            raise ArgumentTypeError("AGD metadata file not present at {}".format(filename))
        with open(filename) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                log.error("Unable to parse AGD metadata file {}".format(filename))
                raise

    parser.add_argument("dataset", type=dataset_parser, help="The AGD json metadata file describing the dataset")
