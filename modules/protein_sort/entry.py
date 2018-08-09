# Modified version of modules/sort/entry.py to sort proteins based on their sequence length
# Protein sort C++ operations are in persona/kernels/sort
# Milad, Aug. 2018

import os
from . import merge_sort
from ..common import service
from common.parse import numeric_min_checker


class LocalSortSingleton(service.ServiceSingleton):
    class_type = merge_sort.LocalSortService


_singletons = [LocalSortSingleton()]
_service_map = {a.get_shortname(): a for a in _singletons}


def get_services():
    return _singletons


def lookup_service(name):
    return _service_map[name]


def get_tooltip():
    return "Sort an AGD protein dataset based on the length of prot column"
