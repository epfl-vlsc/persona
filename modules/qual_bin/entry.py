from . import agd_qual_bin
from ..common import service

def get_tooltip():
  return "Mark PCR duplicate reads in an aligned AGD dataset."

class QualBinSingleton(service.ServiceSingleton):
  class_type = agd_qual_bin.QualBinService


_singletons = [ QualBinSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]
