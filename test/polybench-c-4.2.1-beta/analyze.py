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
from pandas.api.types import CategoricalDtype
from decimal import *

pd.options.display.max_columns = None
# plt.rcParams.update({'font.size': 8})

def safe_div(x,y):
    if y == 0:
        return 0
    return x / y
def main(argv):
    build_dir = argv[0]
    results_file_path = f"{build_dir}/vra.txt"
    results_path = f"{build_dir}/"
    results = pd.read_csv(results_file_path, sep="\s+")
    results['dynamic'] = results.index.str.endswith('-dynamic')
    results['bench'] = results.index.str.extract(r'(?P<benchmark>.*?)(?:-dynamic)?$', expand=False)
    modeDictionary = {True: 'dynamic', False: 'static'}
    results['mode'] = results['dynamic'].map(modeDictionary)
    results['e_perc'] = results['e_perc'].astype(float).abs().div(100)
    # results['e_perc_log'] = np.log10(results['e_perc']).replace(-np.inf, -10).astype(int)
    results['speedup'] = results['speedup'].astype(float).sub(1.0)
    # results.style.format({'e_perc': "{:.2%}",'speedup': "{:.2%}"})
    print(results)

    plot_error(results_path, results)
    plt.show()


def plot_error(results_path, stats_summary_full):
    file_base = f'{results_path}/results'

    fig, axd = plt.subplot_mosaic([['error', 'speedup']], constrained_layout=True)
    fig.suptitle('Error and speedup')
    fig.set_size_inches(22, 12, forward=True)
    float_err = stats_summary_full.pivot(index='bench', columns='mode', values='e_perc')
    sns.heatmap(float_err, annot=True, linewidths=.5, ax=axd['error'], cmap='coolwarm', vmin=0.00001, vmax=0.1, fmt='.5%')
    axd['error'].title.set_text('relative error')
    fixed_err = stats_summary_full.pivot(index='bench', columns='mode', values='speedup')
    sns.heatmap(fixed_err, annot=True, linewidths=.5, ax=axd['speedup'], cmap='coolwarm_r', vmin=-1, vmax=+1, fmt='.2%')
    axd['speedup'].title.set_text('speedup')
    fig.savefig(f'{file_base}.png', dpi=fig.dpi)


if __name__ == "__main__":
    main(sys.argv[1:])