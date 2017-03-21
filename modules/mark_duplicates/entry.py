import os
import argparse
from . import agd_mark_duplicates

def get_tooltip():
  return "Mark PCR duplicate reads in an aligned AGD dataset."

def get_service():
    return agd_mark_duplicates.service()

def get_args(subparser):
  def numeric_min_checker(minimum, message):
      def check_number(n):
          n = int(n)
          if n < minimum:
              raise argparse.ArgumentError("{msg}: got {got}, minimum is {minimum}".format(
                  msg=message, got=n, minimum=minimum
              ))
          return n
      return check_number

  subparser.add_argument("-p", "--parse-parallel", default=1, type=numeric_min_checker(minimum=1, message="read parallelism"),
                      help="total paralellism level for reading data from disk")
  subparser.add_argument("-w", "--write-parallel", default=1, help="number of writers to use",
                      type=numeric_min_checker(minimum=1, message="number of writers min"))

  

