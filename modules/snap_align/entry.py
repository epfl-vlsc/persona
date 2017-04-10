from . import snap_align
from ..common import service

class CephSNAPSingleton(service.ServiceSingleton):
  class_type = snap_align.CephSnapService

class LocalSNAPSingleton(service.ServiceSingleton):
  class_type = snap_align.LocalSnapService

_singletons = [ CephSNAPSingleton(), LocalSNAPSingleton() ]

def get_tooltip():
  return "Alignment using the SNAP aligner"

def get_services():
  return _singletons
