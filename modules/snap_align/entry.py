from . import snap_align
from ..common import service

class CephSNAPSingleton(service.ServiceSingleton):
  class_type = snap_align.CephSnapService

class LocalSNAPSingleton(service.ServiceSingleton):
  class_type = snap_align.LocalSnapService

_singletons = [ CephSNAPSingleton(), LocalSNAPSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_tooltip():
  return "Alignment using the SNAP aligner"

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]