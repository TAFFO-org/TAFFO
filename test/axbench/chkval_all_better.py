#!/usr/bin/env python3

import sys
import re
import os
import statistics


def ParseInput(filename):
  out = []
  with open(filename, 'r') as f:
    f.readline() # discard table header
    l = f.readline()
    while l != '':
      r = re.findall('[a-zA-Z0-9\.+-_]+', l)
      for i in range(1, len(r)):
        if r[i] == '-':
          r[i] = None
        else:
          if i in range(1, len(r)-2):
            r[i] = [float(r[i])]
          else:
            r[i] = float(r[i])
      out.append(r)
      l = f.readline()
  return out

def AppendEverything(big_table, table):
  for i in range(len(big_table)):
    for j in range(1, len(big_table[i])-2):
      if big_table[i][j]:
        big_table[i][j] += table[i][j]

def ComputeAllMedians(t):
  out = []
  for r in t:
    outr = []
    for cell in r:
      if isinstance(cell, list):
        outr.append(statistics.median(cell))
      else:
        outr.append(cell)
    out.append(outr)
  return out
  
def ComputeSpeedup(t):
  for row in t:
    row += [row[2] / row[1]]

def PrettyPrint(table):
  widths=[40,  12,  12,   11,  11, 11,  14, 14]
  format=['s','.6f','.6f','d','d','.5f','.5e','.6f']
  titles=['', 'fix_t', 'flo_t', 'fix_nofl', 'flo_nofl', 'e_perc', 'e_abs', 'speedup']
  normalfmt=['%' + str(widths[i]) + format[i] for i in range(len(widths))]
  fallbackfmt=['%' + str(widths[i]) + 's' for i in range(len(widths))]
  print(''.join([fallbackfmt[i] % titles[i] for i in range(len(widths))]))
  for i in range(len(table)):
    for j in range(len(table[i])):
      if not (table[i][j] is None):
        v = normalfmt[j] % table[i][j]
      else:
        v = fallbackfmt[j] % '-'
      print(v, end='')
    print('')


if len(sys.argv) != 2:
  print("usage: %s <n_tries>" % sys.argv[0])
  exit(1)
ntries = int(sys.argv[1])

t = None
t2 = None
os.system("mkdir -p raw-times")
print("trial 1", file=sys.stderr)
os.system("./chkval_all.sh > raw-times/1.txt")
t = ParseInput("raw-times/1.txt")
for i in range(ntries-1):
  print('trial %d' % (i+2), file=sys.stderr)
  os.system("./chkval_all.sh --noerror > raw-times/%d.txt" % (i+2))
  t2 = ParseInput("raw-times/%d.txt" % (i+2))
  AppendEverything(t, t2)
tmed = ComputeAllMedians(t)
ComputeSpeedup(tmed)
PrettyPrint(tmed)
