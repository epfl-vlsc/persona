import multiprocessing
from . import export_bam
from ..common import service

class ExportBamSingleton(service.ServiceSingleton):
  class_type = export_bam.ExportBamService


_singletons = [ ExportBamSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]

def get_tooltip():
  return "Export AGD dataset to BAM"


