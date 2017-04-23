
from . import fastqimport
from ..common import service

class ImportFastqSingleton(service.ServiceSingleton):
  class_type = fastqimport.ImportFastqService

_singletons = [ ImportFastqSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]

def get_tooltip():
  return "Import FASTQ files into an AGD dataset"

