#!/usr/bin/env python3
__doc__ = '''Load the profile data from multiple files, dump as CSV (standalone) or return Dataframe (as library)'''
from subprocess import getoutput as cmd
import pandas as pd
import numpy as np
import os


def load_file(fname):
	d = pd.read_csv(fname,sep='\s\s+',index_col=0, engine='python')
	names = split(fname[:-4],'-')
	d['Tag']=split(names[0],'/')[-1]
	#print names
	d=d.set_index('Tag',append=True)
	if names[1]=='error' : 
		#print 'dropping times'		
		d=d.drop(columns=['fix T','flo T','# ofl fix', '# ofl flo'])
	elif names[1]=='times' : 
		#print 'dropping errors'	
		d=d.drop(columns=['avg err %','avg abs err','# ofl fix', '# ofl flo'])
	#print d.columns.values
	return d


def load_file_new(fname):
	d = pd.read_fwf(fname, index_col=0)
	tag = os.path.splitext(os.path.basename(fname))[0]
	d['Tag'] = tag
	#print names
	d=d.set_index('Tag',append=True)
	d=d.drop(columns=['fix_nofl', 'flo_nofl', 'speedup'])
	d=d.rename(axis='columns', mapper={'e_abs': 'avg abs err', 'e_perc': 'avg err %', 'fix_t': 'fix T', 'flt_t': 'flo T'})
	#print d.columns.values
	return d
	


def f(*x):
	res = 0.0
	for i in x :
		try :
			res+=float(i)
		except Exception as e :
			return 'drop'
	return res


def load_profile(path='./', boostfail=False):
	file_list = split(cmd('ls '+path+'*-*.txt'))
	if len(file_list) > 0 and os.path.exists(file_list[0]):
		#OLD DIRECTORY FORMAT
		#print file_list
		files = [load_file(fname) for fname in file_list ]
		d=pd.concat(files,axis=1)
	else:
		file_list = split(cmd('ls '+path+'*.txt'))
		files = [load_file_new(fname) for fname in file_list]
		d = pd.concat(files, axis=1)
	cls = d.columns.values
	clss = list(set(cls))

	for x in clss:
		y = '_'.join(split(x))
		d[y] = pd.DataFrame(d[x]).fillna(0).apply(lambda x: f(*x), axis=1)

	d = d.drop(columns=cls)
	#print d
	#print d.columns.values
	return d


if __name__=='__main__':
	d=load_profile('./20190620_polybench/')
	#d = load_profile('./20180911_multiconf_results/')
	print(d)
	d.to_csv(path_or_buf="data.csv")
