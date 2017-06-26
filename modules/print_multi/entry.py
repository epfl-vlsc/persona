from . import agd_print_multi
from ..common import service

def get_tooltip():
  return "Find coverage for an aligned AGD dataset."

class PrintChunkSingleton(service.ServiceSingleton):
  class_type = agd_print_multi.PrintChunkService


_singletons = [ PrintChunkSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]
