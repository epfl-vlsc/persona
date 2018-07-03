from . import agd_protein_cluster
from ..common import service

def get_tooltip():
  return "Cluster and perform all to all protein comparisons"

class ProteinClusterSingleton(service.ServiceSingleton):
  class_type = agd_protein_cluster.ProteinClusterService


_singletons = [ ProteinClusterSingleton() ]
_service_map = { a.get_shortname(): a for a in _singletons }

def get_services():
  return _singletons

def lookup_service(name):
  return _service_map[name]
