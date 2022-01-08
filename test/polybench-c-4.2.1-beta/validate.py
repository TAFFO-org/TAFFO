#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import math
from decimal import *
import argparse
import re
import pandas as pd
from functools import *
import statistics as stat


def PolybenchRootDir() -> Path:
  return Path(os.path.abspath(__file__)).parent


def BenchmarkList():
  blpath = PolybenchRootDir().joinpath(Path('./utilities/benchmark_list'))
  listtext = blpath.read_text()
  res = listtext.strip().split('\n')
  return res


def BenchmarkName(bpath):
  return os.path.basename(os.path.dirname(bpath))


def ReadValues(filename):
  with open(filename, 'r') as f:
    l = f.readline()
    while l != '':
      for v in l.strip().split():
        if v != '':
          yield v
      l = f.readline()


def ComputeDifference(fix_data, flt_data):
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
      
  return {'fix_nofl': fix_nofl, \
          'flo_nofl': flo_nofl, \
          'e_perc': e_perc,
          'e_abs': e_abs}
          
          
def ComputeSpeedups(float_times, fixp_times):
  float_list = [Decimal(di) for di in float_times]
  fixp_list = [Decimal(di) for di in fixp_times]
  float_avg = stat.median(float_list)
  fixp_avg = stat.median(fixp_list)
  speedup = float_avg / fixp_avg
  return {'fix_t': fixp_avg, 'flt_t': float_avg, 'speedup': speedup}
          
          
def PrettyPrint(table):
  df = pd.DataFrame.from_dict(table)
  df = df.transpose()
  return df.to_string()
          

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Validates Polybench output')
  parser.add_argument('--only', dest='only', action='store', default='.*',
                      help='regex of benchmarks to include (default=".*")')
  args = parser.parse_args()

  g_res = {}
  for bench in BenchmarkList():
    if not re.search(args.only, bench):
      continue
    name = BenchmarkName(bench)
    float_dataf = PolybenchRootDir() / 'results-out' / (name+'.float.csv')
    float_data = ReadValues(str(float_dataf))
    float_timesf = PolybenchRootDir() / 'results-out' / (name+'.float.time.txt')
    float_times = ReadValues(str(float_timesf))
    fixp_dataf = PolybenchRootDir() / 'results-out' / (name+'.csv')
    fixp_data = ReadValues(str(fixp_dataf))
    fixp_timesf = PolybenchRootDir() / 'results-out' / (name+'.time.txt')
    fixp_times = ReadValues(str(fixp_timesf))
    try:
      res = ComputeDifference(fixp_data, float_data)
      res.update(ComputeSpeedups(float_times, fixp_times))
      g_res[BenchmarkName(bench)] = res
    except Exception as inst:
      print("Problem With " + name)

    
  print(PrettyPrint(g_res))
    

