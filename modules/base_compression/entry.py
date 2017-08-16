from . import agd_base_compression
from ..common import service

class DisplaySingleton(service.ServiceSingleton):
  class_type = agd_base_compression.DisplayService

_singletons = [ DisplaySingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_tooltip():
  return "Display AGD records on stdout"

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]

