#!/usr/bin/env python3

import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np

def read_run_file(file: Path):
  text = file.read_text()
  benchmarks = text.split('BENCHMARK = ')
  res = {}
  for benchmark in benchmarks[1:]:
    try:
      name, orig, taffo = benchmark.split('\nnumber of platforms')
      gpu_orig = float(re.findall('GPU Time in seconds:\n([^\n]+)\n', orig, re.MULTILINE)[0])
      gpu_taffo = float(re.findall('GPU Time in seconds:\n([^\n]+)\n', taffo, re.MULTILINE)[0])
      res[name] = (gpu_orig, gpu_taffo)
    except Exception:
      name = benchmark.split("\n")[0]
      print(f'error: could not read run bench info of {name} from {file}', file=sys.stderr)
  return res

def read_validate_file(file: Path):
  text = file.read_text()
  benchs = re.findall('^([^ ]+)$', text, re.MULTILINE)
  parts = re.split('^[^ ]+$', text, flags=re.MULTILINE)[1:]
  res = {}
  for bench, part in zip(benchs, parts):
    try:
      e_perc = float(re.findall('^e_perc +([0-9.]+)$', text, re.MULTILINE)[0])
      e_abs = float(re.findall('^e_abs +([0-9.]+)$', text, re.MULTILINE)[0])
      res[bench] = (e_perc, e_abs)
    except Exception:
      print(f'error: could not read run bench info of {bench} from {file}', file=sys.stderr)
  return res

def read_experiment(dir: Path):
  columns = ['bench', 'host_dta', 'kern_arg_dta', 'kern_dta', 't_gpu_orig', 't_gpu_taffo', 'e_perc', 'e_abs']
  table = pd.DataFrame(columns=columns)
  dtas = ['f32', 'f16', 'fixp', 'mixed']
  for host_dta in dtas:
    for kern_arg_dta in dtas:
      for kern_dta in dtas:
        run = dir / (host_dta + '_' + kern_arg_dta + '_' + kern_dta + '_run.txt')
        valid = dir / (host_dta + '_' + kern_arg_dta + '_' + kern_dta + '_validate.txt')
        if not (run.exists() and valid.exists()):
          continue
        data_run = read_run_file(run)
        data_valid = read_validate_file(valid)
        benchs = sorted(list(set(data_run.keys()) | set(data_valid.keys())))
        for bench in benchs:
          try:
            t_gpu_orig, t_gpu_taffo = data_run[bench]
            e_perc, e_abs = data_valid[bench]
            row = [bench, host_dta, kern_arg_dta, kern_dta, t_gpu_orig, t_gpu_taffo, e_perc, e_abs]
            table = pd.concat([table, pd.Series(row, columns).to_frame().T], ignore_index=True)
          except Exception:
            print(f'error: incomplete data for tuple {bench} {host_dta} {kern_arg_dta} {kern_dta}', file=sys.stderr)
  return table

def compute_best_conf(df: pd.DataFrame):
  benchs = sorted(set(df['bench']))
  df['speedup'] = df['t_gpu_orig'] / df['t_gpu_taffo']
  res = None
  for bench in benchs:
    #dfb = df[(df['bench'] == bench) & (df['kern_dta'] == 'fixp') & (df['kern_arg_dta']=='fixp')]
    dfb = df[df['bench'] == bench]
    max_su = max(dfb['speedup'])
    best = dfb[dfb['speedup'] == max_su]
    res = pd.concat([res, best], ignore_index=True)
  return res

def main():
  #print(read_run_file(Path('_experiment_2022-11-02_12-44-12/f32_f16_f16_run.txt')))
  #print(read_validate_file(Path('_experiment_2022-11-02_12-44-12/f32_f16_f16_validate.txt')))
  df = read_experiment(Path('_experiment_2022-11-02_12-44-12'))
  print(df.to_csv())
  print()
  print(compute_best_conf(df).to_csv())

if __name__ == '__main__':
  main()
