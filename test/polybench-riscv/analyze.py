#!/usr/bin/python3

import sys
import pandas as pd
from os import listdir
from os.path import isfile, join, isdir
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = None
# plt.rcParams.update({'font.size': 8})

def main(argv):
    stats_path = "./build_stats"
    stats_summary = pd.DataFrame()
    scale_dirs = [f for f in listdir(stats_path) if isdir(join(stats_path, f))]
    # print(scale_dirs)
    for scale in scale_dirs:
        bench_dirs = [f for f in listdir(join(stats_path, scale)) if isdir(join(stats_path, scale, f))]
        # print(bench_dirs)
        for bench in bench_dirs:
            var_stats_path = join(stats_path, scale, bench, f"{bench}.csv")
            ops_stats_path = join(stats_path, scale, bench, f"{bench}.mix.txt")
            var_stats = pd.read_csv(var_stats_path)
            # print(var_stats)
            ops_stats = pd.read_csv(ops_stats_path, sep=" ", header=None).transpose()
            header = ops_stats.iloc[0]
            ops_stats = ops_stats[1:]
            ops_stats.columns=header
            # print(ops_stats)
            stats_row = {
                'bench': bench,
                'scale': int(scale),
                'var_min': var_stats['var_min'].min(),
                'var_max': var_stats['var_max'].max(),
                'var_isnan': var_stats['var_isnan'].max(),
                'var_isinf': var_stats['var_isinf'].max(),
                'MathOp': ops_stats.iloc[0].get("MathOp", 0),
                'IntegerOp': ops_stats.iloc[0].get("IntegerOp", 0),
                'FloatingPointOp': ops_stats.iloc[0].get("FloatingPointOp", 0),
                'FloatMulDivOp': ops_stats.iloc[0].get("FloatMulDivOp", 0),
                'smul.fix.i32': ops_stats.iloc[0].get("call(llvm.smul.fix.i32)", 0),
                'sdiv.fix.i32': ops_stats.iloc[0].get("call(llvm.sdiv.fix.i32)", 0),
                'CastOp': ops_stats.iloc[0].get("CastOp", 0),
                'Shift': ops_stats.iloc[0].get("Shift", 0),
                'fdiv': ops_stats.iloc[0].get("fdiv", 0),
                'fmul': ops_stats.iloc[0].get("fmul", 0),
            }
            stats_summary = stats_summary.append(stats_row, ignore_index=True)

    stats_summary = stats_summary.sort_values(['bench', 'scale'], ascending=False)
    stats_summary["bench_scale"] = stats_summary['bench'].astype(str) +"-"+ stats_summary["scale"].astype(str)
    print(stats_summary)
    stats_summary.to_csv (f'{stats_path}/stats_summary.csv', index = None, header=True)

    # stats_summary.plot(x='bench_scale',
    #                     y=['smul.fix.i32', 'sdiv.fix.i32', 'fdiv', 'fmul'],
    #                     kind='barh',
    #                     stacked=True,
    #                     ax=ax
    #                    )

    # vmin = min(stats_summary['smul.fix.i32'].values.min(), stats_summary['fmul'].values.min())
    # vmax = max(stats_summary['smul.fix.i32'].values.max(), stats_summary['fmul'].values.max())

    fig, ax = plt.subplots(2, 2, constrained_layout=True)

    smul = stats_summary.pivot(index='bench', columns='scale', values='smul.fix.i32')
    sns.heatmap(smul, annot=True, linewidths=.5, ax=ax[0,0], cmap='Blues')
    ax[0,0].title.set_text('smul.fix.i32')

    fmul = stats_summary.pivot(index='bench', columns='scale', values='fmul')
    sns.heatmap(fmul, annot=True, linewidths=.5, ax=ax[1,0], cmap='Blues')
    ax[1,0].title.set_text('fmul')

    sdiv = stats_summary.pivot(index='bench', columns='scale', values='sdiv.fix.i32')
    sns.heatmap(sdiv, annot=True, linewidths=.5, ax=ax[0,1], cmap='Blues')
    ax[0,1].title.set_text('sdiv.fix.i32')

    fdiv = stats_summary.pivot(index='bench', columns='scale', values='fdiv')
    sns.heatmap(fdiv, annot=True, linewidths=.5, ax=ax[1,1], cmap='Blues')
    ax[1,1].title.set_text('fdiv')


    fig2, ax2 = plt.subplots(2, constrained_layout=True)

    intop = stats_summary.pivot(index='bench', columns='scale', values='IntegerOp')
    sns.heatmap(intop, annot=True, linewidths=.5, ax=ax2[0], cmap='Blues')
    ax2[0].title.set_text('IntegerOp')

    floatop = stats_summary.pivot(index='bench', columns='scale', values='FloatingPointOp')
    sns.heatmap(floatop, annot=True, linewidths=.5, ax=ax2[1], cmap='Blues')
    ax2[1].title.set_text('FloatingPointOp')

    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])