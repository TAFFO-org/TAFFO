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


def list_to_html(lst):
    return "<table><tr><td>" + "</td></tr><tr><td>".join([str(x) for x in lst]) + "</td></tr></table>"
def main(argv):
    build_dir = argv[0]
    configs_file_path = f"{build_dir}/configurations.csv"
    double_configs_file_path = f"{build_dir}/double_configurations.csv"
    results_path = f"{build_dir}/summary"
    configs = pd.read_csv(configs_file_path, sep=",",
                          names=["bench", "arch", "mode", "job_file_base", "stats_job_file_base"])

    static_annots = None
    dynamic_annots = None
    src_lines = None

    for index, config in configs.iterrows():
        bench = config['bench']
        arch = config['arch']
        mode = config['mode']
        if not arch == 'EMBEDDED': continue
        if not bench == '3mm': continue
        if mode == 'float': continue
        job_file_base = config['job_file_base']
        bench_annotations_path = f"{job_file_base}.annotations.csv"
        scr_filename = f"src/{bench}/{bench}.c"
        annots = pd.read_csv(bench_annotations_path, sep=";", header=None,
                    names=["name", "op", "val_min", "val_max", "line", "src_file", "src_path", "implicit_code", "field"])
        src = open(scr_filename, "r")
        src_lines = pd.DataFrame(src, columns=['src_line'])
        src_lines = src_lines.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        src_lines['index_col'] = src_lines.index + 1
        if mode == 'fixed':
            static_annots = annots
        else:
            dynamic_annots = annots

    # print(static_annots)
    # print(dynamic_annots)
    # print(src_lines)

    # all_annots = static_annots.merge(dynamic_annots, on='line', how='outer')
    static_annots = src_lines.merge(
        static_annots, left_on="index_col", right_on="line", how='left'
    ).groupby(['index_col','src_line']).agg({'op': list, 'val_min': list, 'val_max': list, })
    static_annots.reset_index(inplace=True)
    static_annots = static_annots.rename(columns={"op": "static_op", "val_min": "static_val_min", "val_max": "static_val_max"})
    print(static_annots)


    dynamic_annots = src_lines.merge(
        dynamic_annots, left_on="index_col", right_on="line", how='left'
    ).groupby(['index_col','src_line']).agg({'op': list, 'val_min': list, 'val_max': list, })
    dynamic_annots.reset_index(inplace=True)
    dynamic_annots = dynamic_annots.rename(columns={"op": "dynamic_op", "val_min": "dynamic_val_min", "val_max": "dynamic_val_max"})
    print(dynamic_annots)

    all_annots = static_annots.merge(dynamic_annots)
    all_annots = all_annots.filter([
        'static_op', 'static_val_min', 'static_val_max',
        'src_line',
        'dynamic_val_min', 'dynamic_val_max', 'dynamic_op',
    ])

    print(all_annots)
    annots_html = all_annots.to_html(escape=False, index=False, formatters={
        'static_op': list_to_html, 'static_val_min': list_to_html, 'static_val_max': list_to_html,
        'dynamic_val_min': list_to_html, 'dynamic_val_max': list_to_html, 'dynamic_op': list_to_html,
    })
    print(annots_html)
    html_file = open(f'{build_dir}/ranges.html', "w")
    html_file.write(annots_html)
    html_file.close()

    all_annots.to_csv(f'{build_dir}/ranges.csv', index = None, header=True)


if __name__ == "__main__":
    main(sys.argv[1:])