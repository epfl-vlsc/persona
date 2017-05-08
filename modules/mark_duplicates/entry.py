from . import agd_mark_duplicates
from ..common import service

def get_tooltip():
  return "Mark PCR duplicate reads in an aligned AGD dataset."

class MarkDuplicatesSingleton(service.ServiceSingleton):
  class_type = agd_mark_duplicates.MarkDuplicatesService


_singletons = [ MarkDuplicatesSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]
