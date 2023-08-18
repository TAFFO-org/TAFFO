#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import math
import glob
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





def ReadAndComputeDifference(fix_name, flt_name):
  stream = os.popen('./difference.out ' + str(fix_name) + " " + str(flt_name))
  output = stream.read()
  tmp = output.split(",");
  fix_nofl = Decimal(tmp[0].split(":")[1])
  flo_nofl = Decimal(tmp[1].split(":")[1])
  e_perc = Decimal(tmp[2].split(":")[1])
  e_abs = Decimal(tmp[3].split(":")[1])

  return {'fix_nofl': fix_nofl, \
        'flo_nofl': flo_nofl, \
        'e_perc': e_perc,
        'e_abs': e_abs}


  





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



  for resul in glob.glob(r"./result*"):
    results=glob.glob(resul + r"/*")
    print(resul)


    g_res = {}
    for bench in BenchmarkList():
      if not re.search(args.only, bench):
        continue
      name = BenchmarkName(bench)





      partial_result = [x for x in results if name in x]

      partial_result = [x.split(".")[1] for x in partial_result if "float"  not in x]
      partial_result= list(dict.fromkeys(partial_result))

      




      for fixed in partial_result:
        float_result = [x for x in results if name in x]
        float_result = [x for x in float_result if bench.split("/")[-1].split(".")[0] in x]
        float_dataf = PolybenchRootDir() / [x for x in float_result if '.float.csv' in x][0]        
        float_timesf = PolybenchRootDir() / [x for x in float_result if '.float.time.txt' in x][0]





        fixp_dataf = PolybenchRootDir() /  str("."+fixed+'.csv')        
        fixp_timesf = PolybenchRootDir() / str("."+fixed+'.time.txt')


        """
        res = ReadAndComputeDifference(fixp_dataf, float_dataf)
        """

        
        float_data = ReadValues(str(float_dataf))  
        fixp_data = ReadValues(str(fixp_dataf))  
        res = ComputeDifference(fixp_data, float_data)
        

        float_times = ReadValues(str(float_timesf))
        fixp_times = ReadValues(str(fixp_timesf))
        res.update(ComputeSpeedups(float_times, fixp_times))
        g_res[fixed.split("/")[-1]] = res
      
    print(PrettyPrint(g_res))
    

