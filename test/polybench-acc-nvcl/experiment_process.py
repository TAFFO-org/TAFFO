#!/usr/bin/env python3

import sys
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import itertools as it

pd.options.mode.chained_assignment = None

def read_run_file(file: Path):
  text = file.read_text()
  benchmarks = text.split('BENCHMARK = ')
  res = {}
  for benchmark in benchmarks[1:]:
    try:
      name, orig, taffo = benchmark.split('\nnumber of platforms')
      gpu_orig = float(re.findall('GPU Time in seconds:\n([^\n]+)', orig, re.MULTILINE)[0])
      gpu_taffo = float(re.findall('GPU Time in seconds:\n([^\n]+)', taffo, re.MULTILINE)[0])
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

def pareto_bench(df: pd.DataFrame, bench: str):
  df['speedup'] = ((df['t_gpu_orig'] / df['t_gpu_taffo']) - 1) * 100
  df['e_perc'] = df['e_perc'].clip(0, 100)
  df['label'] = df['host_dta'] + '_' + df['kern_arg_dta'] + '_' + df['kern_dta']

  def xkey(df: pd.DataFrame):
    return df['speedup']
  def ykey(df: pd.DataFrame):
    return -df['e_perc']
  frontier = set()
  def dominates(a, b): # True if b dominates a
    xa, ya = xkey(df)[a], ykey(df)[a]
    xb, yb = xkey(df)[b], ykey(df)[b]
    return xa <= xb and ya <= yb
  for i, row in df.iterrows():
    if any([dominates(i, j) for j in frontier]):
      continue
    domset = set([j for j in frontier if dominates(j, i)])
    frontier = (frontier - domset) | {i}
  infrontier = df[[i in frontier for i in df.index]].sort_values(by='speedup')
  notfrontier = df[[i not in frontier for i in df.index]]

  objs = []
  objs += [plt.gca().scatter(notfrontier['speedup'], notfrontier['e_perc'], marker='+')]
  plt.gca().plot(infrontier['speedup'], infrontier['e_perc'], "+-", color='orange')
  plt.gca().axvline(0, color='black', linewidth=0.5)
  plt.gca().axhline(0, color='black', linewidth=0.5)
  text = []
  for i, row in infrontier.iterrows():
    text += [plt.gca().text(row['speedup'], row['e_perc'], row['label'], fontsize='small', ha='left', va='bottom')]
  plt.autoscale()
  plt.gca().set_xlim(tuple(map(sum, zip(plt.gca().get_xlim(), (-25, 25)))))
  plt.gca().set_ylim(tuple(map(sum, zip(plt.gca().get_ylim(), (-25, 25)))))
  adjust_text(text, arrowprops=dict(arrowstyle="-", color='k'), autoalign=False)
  #plt.xlabel('Speedup [%]')
  #plt.ylabel('Percentage error [%]')
  plt.gca().invert_yaxis()
  #plt.gca().grid(visible=True, which='both')
  plt.gca().set_title(bench)

def pareto(df: pd.DataFrame):
  benchs = sorted(set(df['bench']))
  fig, ax = plt.subplots(2, 3)
  fig.set_figwidth(15)
  fig.set_figheight(7)
  axes_index = it.product(range(0,3), range(0,3))
  for bench, (i, j) in zip(benchs, axes_index):
    plt.sca(ax[i, j])
    #plt.gca().set_xlim(-150, 250)
    #plt.gca().set_ylim(-25, 125)
    pareto_bench(df[df['bench'] == bench], bench)
  fig.supxlabel('Speedup [%]')
  fig.supylabel('Percentage error [%]')
  fig.tight_layout()
  plt.show()
  #fig.savefig("out.pdf")

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
  df = read_experiment(Path('_experiments/2023-02-21_20-09-32'))
  #print(df.to_csv())
  #print()
  #print(compute_best_conf(df).to_csv())
  pareto(df)

if __name__ == '__main__':
  main()
