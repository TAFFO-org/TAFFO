#!/usr/bin/python

import sys
import math


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def printUsage():
	print "Usage: python qos.py <original file> <nn file>"
	exit(1)
pass;


if(len(sys.argv) != 3):
	printUsage()

origFilename 	= sys.argv[1]
nnFilename		= sys.argv[2]

origLines 		= open(origFilename).readlines()
nnLines			= open(nnFilename).readlines()


missPred = 0
absError = 0.0

for i in range(len(origLines)):

	origLine 	= origLines[i].rstrip()
	nnLine 		= nnLines[i].rstrip()

	origItem 	= int(origLine.split(" ")[0])
	nnItem 		= int(nnLine.split(" ")[0])
        
        origIsect0      = float(origLine.split(" ")[1])
        origIsect1      = float(origLine.split(" ")[2])

        nnIsect0        = float(nnLine.split(" ")[1])
        nnIsect1        = float(nnLine.split(" ")[2])

	if(origItem != nnItem):
		missPred += 1

        diff0 = origIsect0 - nnIsect0
        diff1 = origIsect1 - nnIsect1
        absError += diff0*diff0 + diff1*diff1

pass;

print bcolors.WARNING	+ "*** Absolute error: %d" % (missPred) + bcolors.ENDC
print bcolors.WARNING	+ "*** Relative error: %1.8f %%" % (missPred/float(len(origLines)) * 100.0) + bcolors.ENDC
print bcolors.WARNING	+ "*** Absolute intermediate error: %1.8f" % math.sqrt(absError/float(len(origLines)*2)) + bcolors.ENDC
