from . import agd_peak_detection
from ..common import service

def get_tooltip():
  return "Find peak regions in an AGD dataset."

class CalculateCoverageSingleton(service.ServiceSingleton):
  class_type = agd_peak_detection.CalculateCoverageService


_singletons = [ CalculateCoverageSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]
