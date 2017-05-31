
from . import bamimport
from ..common import service

class ImportBamSingleton(service.ServiceSingleton):
  class_type = bamimport.ImportBamService

_singletons = [ ImportBamSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]

def get_tooltip():
  return "Import BAM file into an AGD dataset"

