#!/usr/bin/env python3

from decimal import *
import argparse
import pandas as pd


def file_reader(filename):
  with open(filename, 'r') as f:
    l = f.readline()
    while l != '':
      for v in l.strip().split():
        if v != '':
          yield v
      l = f.readline()


def compute_difference(fix_data, flt_data):
  n = 0
  accerr = Decimal(0)
  accval = Decimal(0)
  fix_nofl = 0
  flo_nofl = 0

  thres_ofl_cp = Decimal('0.01')

  for svfix, svflo in zip(fix_data, flt_data):
    vfix, vflo = Decimal(svfix), Decimal(svflo)

    if vfix.is_nan():
      fix_nofl += 1
    elif vflo.is_nan():
      flo_nofl += 1
      fix_nofl += 1
    elif ((vflo + vfix).copy_abs() - (vflo.copy_abs() + vfix.copy_abs())) > thres_ofl_cp:
      fix_nofl += 1
    else:
      n += 1
      accerr += (vflo - vfix).copy_abs()
      accval += vflo
      
  e_perc = (accerr / accval * 100) if accval > 0 and n > 0 else -1
  e_abs = (accerr / n) if n > 0 else -1
      
  return {'taffo_overflows': fix_nofl, \
          'normal_overflows': flo_nofl, \
          'e_perc': e_perc,
          'e_abs': e_abs}


def main():
  parser = argparse.ArgumentParser(description='Validates Polybench output')
  parser.add_argument('--ref', dest='ref', action='store', help='Reference file')
  parser.add_argument('--check', dest='check', action='store', help='File to check')
  args = parser.parse_args()

  ref_data = file_reader(args.ref)
  check_data = file_reader(args.check)
  try:
    res = compute_difference(check_data, ref_data)
  except Exception as inst:
    print("Problem with the file")
  print('\n'.join([('%-20s ' % label) + str(value) for label, value in res.items()]))


if __name__ == "__main__":
  main()

