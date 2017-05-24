from . import agd_gene_coverage
from ..common import service

def get_tooltip():
  return "Find coverage for an aligned AGD dataset."

class CalculateCoverageSingleton(service.ServiceSingleton):
  class_type = agd_gene_coverage.CalculateCoverageService


_singletons = [ CalculateCoverageSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]
