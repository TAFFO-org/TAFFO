#!/usr/bin/env python3
__doc__ = '''Load the compiler stats data from multiple files, dump as CSV (standalone) or return Dataframe (as library)'''
from subprocess import getoutput as cmd
import pandas as pd
import numpy as np
import os
import sys


def load_file_stats(fname):
	names = split(fname, '/')
	names = split(names[-1], '_')[0]
	tag = 'good' if 'good' in fname else ('bad' if 'bad' in fname else 'worse')
	out = True if 'fix' in fname else False
	return load_one_stats(fname, tag, names, out)


def load_one_stats(stm_or_fname, tag, bench, fixed):
	d = pd.read_csv(stm_or_fname, sep='\s+', index_col=0, header=None, engine='python')
	d=d.transpose()
	d['Tag']= tag
	d['Bench']=bench
	d['Out'] = 'fix' if fixed else 'float'
	d=d.set_index('Tag',append=False)
	d=d.set_index('Bench',append=True)
	d=d.set_index('Out',append=True)
	return d


def load_file_stats_new(fname):
	names = split(fname, '/')
	names = split(names[-1], '.')[0]
	tag = os.path.basename(os.path.dirname(fname))
	out = False if 'float' in fname else True
	return load_one_stats_new(fname, tag, names, out)


def load_one_stats_new(stm_or_fname, tag, bench, fixed):
	d = pd.read_csv(stm_or_fname,sep='\s+', index_col=0, header=None, engine='python')
	d=d.transpose()
	d['Tag']= tag
	d['Bench']=bench
	d['Out'] = 'fix' if fixed else 'float'
	d=d.set_index('Tag',append=False)
	d=d.set_index('Bench',append=True)
	d=d.set_index('Out',append=True)
	return d


def load_stats(path_or_data='./'):
	if isinstance(path_or_data, str):
		path = path_or_data
		file_list = split(cmd('ls '+path+'*-*/*_ic_*'))
		if len(file_list) > 0 and os.path.exists(file_list[0]):
			# OLD DIRECTORY FORMAT
			file_list = [ x for x in file_list if 'raw' not in x ]
			files = [load_file_stats(fname) for fname in file_list ]
		else:
			file_list = split(cmd('ls ' + path + '*/*.mlfeat.txt'))
			files = [load_file_stats_new(fname) for fname in file_list]
	else:
		files = path_or_data
	d = pd.concat(files)
	d = d.unstack(level='Out')
	d = d.swaplevel('Tag','Bench')
	return d


if __name__=='__main__':
	#d=load_stats('./20180911_multiconf_results/')
	d = load_stats('./20190620_polybench/')
	print(d, file=sys.stderr)
	d.to_csv(path_or_buf="stats.csv")
