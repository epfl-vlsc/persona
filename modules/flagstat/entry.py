from . import agd_flagstat
from ..common import service

def get_tooltip():
  return "Flagstat module that gathers and displays stats on a dataset"

class FlagstatSingleton(service.ServiceSingleton):
  class_type = agd_flagstat.FlagstatService


_singletons = [ FlagstatSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]
