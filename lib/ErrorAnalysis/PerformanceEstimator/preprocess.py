#!/usr/bin/env python3
import pandas as pd
from stats_loader import load_stats
from profile_loader import load_profile


def preprocess_features_instfrequency(orig_feat, d3):
	new_features = []
	for f in orig_feat:
		a, b = f
		if a[0] != 'B':
			if b == 'fix':
				f1 = (a, 'float')
				d3[a] = (d3[f] - d3[f1]) / d3[('*', 'float')].astype(float)
				if d3[a].nunique() > 1:
					new_features.append(a)
		else:
			parts = a.split('_')
			if parts[1] != 'n':
				new_features.append(a)
			if b == 'fix':
				f1 = (a, 'float')
				n = d3[(parts[0] + '_n_*', 'float')].fillna(1).replace(0, 1).astype(float)
				d3[a] = (d3[f] - d3[f1]) / n
				if d3[a].nunique() > 1:
					new_features.append(a)
	return new_features


def preprocess_features_freqandtrim(orig_feat, d3):
	nblock_threshold = 2
	new_features = []
	for f in orig_feat:
		a, b = f
		parts = a.split('_')
		if len(parts) == 3 and parts[1] == 'contain':
			nblock2 = int(parts[2][1:])
			if nblock2 > nblock_threshold:
				continue
		nblock = int(parts[0][1:])
		if nblock > nblock_threshold:
			continue
		if parts[1] == 'minDist':
			continue
		elif parts[1] != 'n':
			if b == 'fix':
				new_features += [a]
				d3[a] = d3[(a, b)]
		else:
			if not parts[2][0].isupper():
				continue
			if parts[2][0:5] == 'call(':
				continue
			if parts[2] == '*':
				continue
			new_features += [b+'_'+a]
			d3[b+'_'+a] = d3[(a,b)] / d3[(parts[0] + '_n_*', b)].replace(0, 1).astype(float)
	return new_features


def load_data_phase1(path_or_data, fpreprocess=None):
	if isinstance(path_or_data, str):
		d1=load_stats(path_or_data)
		d2=load_profile(path_or_data)
		d3=pd.concat([d1,d2],axis=1)
		d3 = d3.drop('durbin', axis=0, errors='ignore')
	else:
		d1=load_stats(path_or_data)
		d3=d1.copy()
		d3.columns = d3.columns.to_flat_index()
	d3=d3.fillna(0)

	if not fpreprocess:
		fpreprocess = preprocess_features_instfrequency
		#fpreprocess = preprocess_features_freqandtrim
	new_features = fpreprocess(d1.columns.values, d3)
	return new_features, d3

