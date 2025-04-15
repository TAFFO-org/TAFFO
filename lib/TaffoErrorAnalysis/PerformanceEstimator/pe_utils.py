#!/usr/bin/env python3


def median(iter):
	iter.sort()
	if len(iter) % 2 == 1:
		return iter[len(iter)/2]
	else:
		return (iter[len(iter)/2] + iter[len(iter)/2+1]) / 2
