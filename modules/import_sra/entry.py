
from . import sraimport
from ..common import service

class ImportSraSingleton(service.ServiceSingleton):
  class_type = sraimport.ImportSraService

_singletons = [ ImportSraSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]

def get_tooltip():
  return "Import SRA file into an AGD dataset"

