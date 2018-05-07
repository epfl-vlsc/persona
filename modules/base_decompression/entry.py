from . import agd_base_decompression
from ..common import service

class BaseDecompressionSingleton(service.ServiceSingleton):
  class_type = agd_base_decompression.AGDBaseDecompressionService

_singletons = [ BaseDecompressionSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_tooltip():
  return "Display AGD records on stdout"

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]
