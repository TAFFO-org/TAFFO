#!/usr/bin/env python3
import pandas as pd
import numpy as np
from math import sqrt
from stats_loader import load_stats
from profile_loader import load_profile
import time as pytime
import argparse
from preprocess import *
import sys


better_than = 0.2  # 1=go fixed if time is less than float, 0.5 if time is half then float, etc.


def load_data_onlystats(path, features=None):
	new_features, d3 = load_data_phase1(path)

	if not features:
		features = new_features

	print(features, file=sys.stderr)
	for f in features:
		if f not in d3.columns.values:
			d3[f] = 0.

	print(d3.loc[:, features].shape, file=sys.stderr)
	return d3.fillna(0), features


def load_data(path, features=None, response=None, boostfail=0):
	d3, features = load_data_onlystats(path, features)

	d3['ratio'] = d3['flo_T'] / d3['fix_T']
	w1 = d3['fix_T'] < (d3['flo_T'] * better_than)
	w2 = d3['fix_T'] < (d3['flo_T'])
	# d3['worth']=w1.astype(int)+w2.astype(int)-1
	# d3['worth']=d3['fix_T'] < (d3['flo_T'])
	d3['worth'] = ((d3['ratio'] - 1.0) / 0.2).astype(int).clip(-1, 1)
	# d3['worth'] = w1

	if boostfail > 0:
		for rowi in d3.index:
			if int(d3.loc[rowi, 'worth']) < 1:
				row = d3.loc[rowi, :]
				for repi in range(0, boostfail):
					cprow = row.copy()
					cprow.name = (cprow.name[0] + '_c' + str(repi), cprow.name[1])
					d3 = d3.append(cprow)

	# print d3['ratio'], d3['worth']
	print('<20%', sum(w1), 'of', len(d3['worth']))
	print('<100%', sum(w2) - sum(w1), 'of', len(d3['worth']))
	print('>100%', len(d3['worth']) - sum(w2), 'of', len(d3['worth']))

	if not response:
		response = ['worth']

	print(d3.loc[:, response].shape, file=sys.stderr)
	return d3.fillna(0), features, response


def split_train_test(d3):
	msk = [n < 0.8 * len(d3) for n in range(0, len(d3))]
	np.random.shuffle(msk)
	train = d3[msk]
	test = d3[[not v for v in msk]]
	return train, test


def do_test_prediction(fitr, test, features, response, oneatatime):
	if not oneatatime:
		benchs = [test.index]
	else:
		benchs = [[x] for x in test.index]
	results, times = [], []
	for bench in benchs:
		res = fitr.score(test.loc[bench, features], np.ravel(test.loc[bench, response]))
		t0 = pytime.time()
		pred = fitr.predict(test.loc[bench, features])
		t1 = pytime.time()
		print('Predict', bench, pred)
		print('Actual', np.ravel(test.loc[bench, response]))
		results += [res]
		times += [t1 - t0]
	return results, times


def train_adaboost(train, features, response):
	from sklearn.ensemble import AdaBoostClassifier
	regr = AdaBoostClassifier(n_estimators=100)
	fitr = regr.fit(train.loc[:, features], np.ravel(train.loc[:, response]))
	return fitr


def train_gradient(train, features, response):
	from sklearn.ensemble import GradientBoostingClassifier
	regr = GradientBoostingClassifier(max_features=int(sqrt(len(features))))
	fitr = regr.fit(train.loc[:, features], np.ravel(train.loc[:, response]))
	return fitr


def train_extree(train, features, response):
	from sklearn.ensemble import ExtraTreesClassifier
	regr = ExtraTreesClassifier(max_features=int(sqrt(len(features))))
	fitr = regr.fit(train.loc[:, features], np.ravel(train.loc[:, response]))
	return fitr


def train_bagging(train, features, response):
	from sklearn.ensemble import BaggingClassifier
	regr = BaggingClassifier(max_features=int(sqrt(len(features))))
	fitr = regr.fit(train.loc[:, features], np.ravel(train.loc[:, response]))
	return fitr


def train_randomforest(train, features, response):
	from sklearn.ensemble import RandomForestClassifier
	regr = RandomForestClassifier(max_features=int(sqrt(len(features))))
	fitr = regr.fit(train.loc[:, features], np.ravel(train.loc[:, response]))
	return fitr


def train_multilayerperceptron(train, features, response):
	from sklearn.neural_network import MLPClassifier
	regr = MLPClassifier(solver='lbfgs', alpha=1e-5,
						 hidden_layer_sizes=(int(len(features) / 2), int(len(features) / 8)), random_state=1)
	fitr = regr.fit(train.loc[:, features], np.ravel(train.loc[:, response]))
	return fitr


def train_svc(train, features, response):
	from sklearn.svm import SVC
	regr = SVC()
	fitr = regr.fit(train.loc[:, features], np.ravel(train.loc[:, response]))
	return fitr

