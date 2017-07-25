from . import import_sga
from ..common import service

def get_tooltip():
  return "Convert AGD dataset to SGA."

class CalculateCoverageSingleton(service.ServiceSingleton):
  class_type = import_sga.CalculateCoverageService


_singletons = [ CalculateCoverageSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]
