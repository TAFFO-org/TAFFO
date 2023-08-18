#!/usr/bin/env python3

from decimal import *
import argparse
import math


def file_reader(filename):
  with open(filename, 'r') as f:
    l = f.readline()
    while l != '':
      for v in l.strip().split():
        if v != '':
          yield v
      l = f.readline()


def compute_difference(fix_data, flt_data):
  vmax = -float("inf")
  vmin = float("inf")
  emax = -float("inf")
  for svfix, svflo in zip(fix_data, flt_data):
    e = abs(float(svfix) - float(svflo))
    print(e, svflo, svfix)
    vmax = max(vmax, float(svflo))
    vmin = min(vmin, float(svflo))
    emax = max(emax, e)
  print("float max =", vmax)
  print("float min =", vmin)
  print("diff max  =", emax)


def main():
  parser = argparse.ArgumentParser(description='Validates Polybench output')
  parser.add_argument('--ref', dest='ref', action='store', help='Reference file')
  parser.add_argument('--check', dest='check', action='store', help='File to check')
  args = parser.parse_args()

  ref_data = file_reader(args.ref)
  check_data = file_reader(args.check)
  compute_difference(check_data, ref_data)


if __name__ == "__main__":
  main()

