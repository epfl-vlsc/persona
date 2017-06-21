from . import agd_stat
from ..common import service

def get_tooltip():
  return "Stat module that displays information about an AGD dataset."

class StatSingleton(service.ServiceSingleton):
  class_type = agd_stat.StatService


_singletons = [ StatSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]
