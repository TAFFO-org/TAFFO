#!/usr/bin/env python3

import sys
from pathlib import Path, PosixPath
from decimal import *
import functools


def read_data(path: Path):
  if not path.exists():
    return None
  lines = path.read_text().splitlines()
  numbers = functools.reduce(lambda l1, l2: l1+l2, [l.split() for l in lines], [])
  return [l for l in [l.strip() for l in numbers] if l != ""]


def compute_error(fix_data, flt_data):
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
      
  return {'fix_nofl': fix_nofl,
          'flo_nofl': flo_nofl,
          'e_perc': e_perc,
          'e_abs': e_abs}


def main():
  if len(sys.argv) < 3:
    print("error: expected two arguments")
    return 1
  flt, fix = Path(sys.argv[1]), Path(sys.argv[2])
  flt_data, fix_data = read_data(flt), read_data(fix)
  if fix_data is None or flt_data is None or len(fix_data)==0 or len(flt_data)==0:
    print('error: no data')
    return 1
  errs = compute_error(fix_data, flt_data)
  print('{: .8e} {: .8e}'.format(errs['e_perc'], errs['e_abs']))


if __name__ == "__main__":
  exit(main())
    

