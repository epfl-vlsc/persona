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

def dump_metadata(mdata):
    if 'name' in mdata:
        print("AGD Dataset: {}".format(mdata['name']))
        print("AGD version: {}".format(mdata['version']))
    if 'columns' in mdata:
        print("Columns present:")
        for c in mdata['columns']:
            print(c)
    else:
        print("No columns found, your dataset may be corrupted?")
    if 'records' in mdata:
        recs = mdata['records']
        print("{} chunks, {} files in total.".format(len(recs), len(mdata['columns'])*len(recs)))
        chunk_size = recs[0]['last']
        print("Chunk size: {}".format(chunk_size))
        num_rows = chunk_size*(len(recs) - 1)
        num_rows += recs[-1]['last'] - recs[-1]['first']
        print("Total records: {}".format(num_rows))

    if 'sort' in mdata:
        print("Sort order: {}".format(mdata['sort']))
    
    if 'reference' in mdata:
        print("Aligned to reference: {}".format(mdata['reference']))



   

