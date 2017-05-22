from ..common import service
from . import services

class EchoSingleton(service.ServiceSingleton):
    class_type = services.EchoService

class IncrSingleton(service.ServiceSingleton):
    class_type = services.Incrementer

_singletons = (EchoSingleton(), IncrSingleton())
_service_map = { a.get_shortname(): a for a in _singletons }

def get_tooltip():
    return "simple services to use for debugging"

def get_services():
    return _singletons

def lookup_service(name):
    return _service_map[name]
