#!/usr/bin/env python3

from glob import glob
from locale import normalize
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
import re 
import rename

def swapPositions(list, pos1, pos2):
     
    # popping both the elements from list
    first_ele = list.pop(pos1)  
    second_ele = list.pop(pos2-1)
    
    # inserting in each others positions
    list.insert(pos1, second_ele) 
    list.insert(pos2, first_ele) 
     
    return list


def ReadCycles(string):
    return re.findall(r"(\S*) cycles",string)

def ReadValues(string):
    cap = re.findall(r"Values Start([\S\s]*)Values End", string)
    return  re.findall(r'-?[0-9]+(?:\.[0-9]+)?',cap[0])

def ReadFile(path):
    with open(path,"r") as file:
        return file.read()



def ComputeDifference(norm_data, other_data):
  n = 0
  accerr = Decimal(0)
  accval = Decimal(0)
  oth_nofl = 0
  nor_nofl = 0

  thres_ofl_cp = Decimal('0.01')

  for svoth, svnor in zip(other_data, norm_data):
    if svoth.is_nan():
      oth_nofl += 1
    elif svnor.is_nan():
      nor_nofl += 1
      oth_nofl += 1
    elif ((svnor + svoth).copy_abs() - (svnor.copy_abs() + svoth.copy_abs())) > thres_ofl_cp:
      oth_nofl += 1
    else:
      n += 1
      accerr += (svnor - svoth).copy_abs()
      accval += svnor
      
  e_perc = (accerr / accval * 100) if accval > 0 and n > 0 else -1
  e_abs = (accerr / n) if n > 0 else -1
      
  return {
          'e_perc': e_perc,
          'e_abs': e_abs}
          
          
def ComputeSpeedups(norm_times, other_times):
  norm_avg = stat.mean(norm_times)
  other_avg = stat.mean(other_times)
  speedup = norm_avg / other_avg
  return {'other_t': other_avg, 'norm_t': norm_avg, 'speedup': speedup}
          
          
def PrettyPrint(rows):
  df = pd.DataFrame(data=rows)
  print(df.to_string())

def LatexMe(rows):
  df = pd.DataFrame(data=rows)
  df = df.sort_values(by="name", ascending=False)
  callable = {
    "name" : "{}",
    "e_perc" : "{0:.9f}",
    "e_abs" : "{0:.9f}",
    "speedup" : "{0:.9f}"
  }
  string = \
r"""\begin{table}[h]
\centering
\small
"""
  string += df.style.format(callable).to_latex()
  string += \
r"""\end{table}
"""  
  string += "\n\\clearpage\n"
  return string

def LoadBench(file_path: str, bench_dict):
        bench_name = file_path.split("/")[-1]
        all_file = ReadFile(file_path)          
        cycle = [Decimal(x) for x in ReadCycles(all_file)]
        values = [Decimal(x) for x in ReadValues(all_file)]
        bench_dict[bench_name] = {"time" : cycle, "data" : values}      

def NormalizeOn(norm, bench_dict):
    rows = []
    if "float" not in norm:
          return
    for other in bench_dict:
        if norm == other:
            continue

        speedup = ComputeSpeedups(bench_dict[norm]["time"], bench_dict[other]["time"])
        difference = ComputeDifference(bench_dict[norm]["data"], bench_dict[other]["data"])
        row = {"name" : other }
        row.update(difference)
        row["speedup"] = speedup["speedup"]
        rows.append(row)
    string ="\nNormalize on \\textbf{{{}}}\n\n".format(norm, bench_dict[norm]["time"])
    string += LatexMe(rows)
    bench_dict[norm]["latex"] = string



def PrintLatex(total_bench):
  print("\n\\subsection*{Norm to Float}\n\n")
  for bench_dict in total_bench:
    for bench in bench_dict:
      if "float" in bench:
        print(bench_dict[bench]["latex"].replace("_","\\_"))


    
  # print("\n\\subsection*{All the other}\n\n")
  # for bench_dict in total_bench:
    # for bench in bench_dict:
      # if "float" not in bench:
        # print(bench_dict[bench]["latex"].replace("_","\\_"))

  print("\n\n\\subsection*{Times:} \n")
  for bench_dict in total_bench:
    print("\\vspace{5mm}")
    for bench in bench_dict:
        print("\\textbf{{{}}}\t$\\Rightarrow$\t {}\n".format(bench.replace("_","\\_"), stat.mean(bench_dict[bench]["time"])))
    print("\n")




if __name__ == "__main__":
  rename.rename()
  full_path = os.path.dirname(os.path.realpath(__file__))
  total_bench = []
  for bench in glob(full_path + "/output/*"):   
    bench_dict = {}
    """"Load Taffo Bench"""
    for file_path in [ x for x in glob( bench+"/taffo/*.log") if x.split("/")[-1].find("bench") == -1 ]:
        try:
          LoadBench(file_path, bench_dict)
        except:
          print("Crash processing:")
          print(file_path)
          exit()

    
    """Load Float Bench"""   
    for file_path in [ x for x in glob( bench+"/float/*.log") if x.split("/")[-1].find("bench") == -1 ]:
        LoadBench(file_path, bench_dict)

    for norm in bench_dict:
            NormalizeOn(norm, bench_dict)
    total_bench.append(bench_dict)


  PrintLatex(total_bench)
  

        

    

