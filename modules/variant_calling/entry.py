from . import variant_calling
from ..common import service

def get_tooltip():
  return "Variant Calling for AGD Dataset"

class PrintChunkSingleton(service.ServiceSingleton):
  class_type = variant_calling.PrintChunkService


_singletons = [ PrintChunkSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]
