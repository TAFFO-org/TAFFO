#!/usr/bin/python3

import sys
import pandas as pd

def main(argv):
    inputfile = argv[0]
    df = pd.read_csv(inputfile)
    grouped = df.groupby('var_name')
    minVars = grouped['var_min'].min()
    maxVars = grouped['var_max'].max()
    for i, v in minVars.items():
        print(f"-DVAR_{i}_MIN={v} ", end="")

    for i, v in maxVars.items():
        print(f"-DVAR_{i}_MAX={v} ", end="")

if __name__ == "__main__":
    main(sys.argv[1:])