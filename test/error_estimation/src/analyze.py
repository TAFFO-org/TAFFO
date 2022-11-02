#!/usr/bin/python3

import sys
import pandas as pd
from os import listdir
from os.path import isfile, join, isdir
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO
import math

pd.options.display.max_columns = None
# plt.rcParams.update({'font.size': 8})

def safe_div(x,y):
    if y == 0:
        return 0
    return x / y


def main(argv):
    f = open('build/data.txt', 'r')
    exact = float(f.readline())

    print(f"{exact}")
    df = pd.read_csv('build/summary.csv', sep=",", header=None)
    df.columns = ['experiment', 'result']
    df['result'] = df['result'].astype(float)
    df['mantissa'] = df['experiment'].str.extract(r'-m(\d+)bit-')
    df['mantissa'] = df['mantissa'].astype(int)
    df['input_size'] = df['experiment'].str.extract(r'-input(\d+)rows')
    df['input_size'] = df['input_size'].astype(int)
    df['abs_err'] = abs(exact - df['result'])
    df['rel_err'] = df['abs_err'] / df['result']
    df['log2_err'] = df['rel_err'].apply(np.log2).apply(np.ceil)
    df['predicted_log2_err'] = df['input_size'].apply(np.log2).apply(np.ceil) - df['mantissa']

    print(df)
    df.to_csv (f'build/stats_summary.csv', index = None, header=True)

    fig, ax = plt.subplots(2, 2, constrained_layout=True)
    fig.set_size_inches(12, 12, forward=True)

    log2_err = df.pivot(index='input_size', columns='mantissa', values='log2_err')
    log2_err.plot(ax=ax[0,0], marker='o')
    ax[0,0].title.set_text('measured log2(err)')
    ax[0,0].grid()
    ax[0,0].locator_params(axis="both", integer=True, tight=True)
    ax[0,0].ticklabel_format(useOffset=False, style='plain')
    ax[0,0].set_xscale('log')
    ax[0,0].set_ylim([-25, 15])

    pred_log2_err = df.pivot(index='input_size', columns='mantissa', values='predicted_log2_err')
    pred_log2_err.plot(ax=ax[1,0], marker='o')
    ax[1,0].title.set_text('predicted log2(err)')
    ax[1,0].grid()
    ax[1,0].locator_params(axis="both", integer=True, tight=True)
    ax[1,0].ticklabel_format(useOffset=False, style='plain')
    ax[1,0].set_xscale('log')
    ax[1,0].set_ylim([-25, 15])

    log2_err_tbl = df.pivot(index='mantissa', columns='input_size', values='log2_err')
    sns.heatmap(log2_err_tbl, annot=True, linewidths=.5, ax=ax[0,1], cmap='coolwarm', vmin=-25, vmax=25)
    ax[0,1].title.set_text('measured log2(err)')

    pred_log2_err_tbl = df.pivot(index='mantissa', columns='input_size', values='predicted_log2_err')
    sns.heatmap(pred_log2_err_tbl, annot=True, linewidths=.5, ax=ax[1,1], cmap='coolwarm', vmin=-25, vmax=25)
    ax[1,1].title.set_text('predicted log2(err)')

    fig.savefig(f'build/err.png', dpi=fig.dpi)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])



