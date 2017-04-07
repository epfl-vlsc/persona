from . import snap_align
from ..common import parse, service

class CephSNAPSingleton(service.ServiceSingleton):
  class_type = snap_align.CephSnapService

_singletons = [ CephSNAPSingleton() ]

def get_tooltip():
  return "Alignment using the SNAP aligner"

def get_services():
  return _singletons
