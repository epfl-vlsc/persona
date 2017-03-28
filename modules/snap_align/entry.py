import argparse
import multiprocessing
import os
import getpass
from . import snap_align
from ..common import parse, service

class SNAPSingleton(service.ServiceSingleton):
  class_type = snap_align.SnapService

def get_tooltip():
  return "Perform single or paired-end alignment on an AGD dataset using SNAP"

def get_service():
  return SNAPSingleton()
