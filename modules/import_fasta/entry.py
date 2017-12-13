
from . import fastaimport
from ..common import service

class ImportFastaSingleton(service.ServiceSingleton):
  class_type = fastaimport.ImportFastaService

_singletons = [ ImportFastaSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]

def get_tooltip():
  return "Import FASTA files into an AGD dataset"

