import multiprocessing
from . import filtering
from ..common import service

class FilteringSingleton(service.ServiceSingleton):
  class_type = filtering.FilteringService


_singletons = [ FilteringSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]

def get_tooltip():
  return "Filter dataset based on given predicate."


