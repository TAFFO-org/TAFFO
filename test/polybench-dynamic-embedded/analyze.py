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
    configs_file_path = f"{build_dir}/configurations.csv"
    double_configs_file_path = f"{build_dir}/double_configurations.csv"
    results_path = f"{build_dir}/summary"
    configs = pd.read_csv(configs_file_path, sep=",",
                          names=["bench", "arch", "mode", "job_file_base", "stats_job_file_base"])
    double_configs = pd.read_csv(double_configs_file_path, sep=",",
                                 names=["bench", "job_file_base"])
    print(configs)
    stats_summary = pd.DataFrame()
    bench_results = pd.DataFrame()
    highest_precision = pd.DataFrame()
    for index, config in double_configs.iterrows():
        bench = config['bench']
        job_file_base = config['job_file_base']
        bench_results_path = f"{job_file_base}.csv"
        bench_times_path = f"{job_file_base}.time.txt"

        try:
            regular_data = np.fromiter(ReadValues(bench_results_path), dtype=object, count=-1)
            avg_time = read_time(bench_times_path).mean()['time_var']
            data_row1 = {
                'bench': bench,
                'job_file_base': job_file_base,
                'bench_type': 'double',
                'data': regular_data,
                'avg_time': avg_time
            }
            row_df = pd.DataFrame([data_row1])
            highest_precision = pd.concat([highest_precision, row_df], ignore_index=True)
        except Exception as inst:
            print(f"{bench}_PC_double ops stats compilation error: {str(inst)}")

    for index, config in configs.iterrows():
        bench = config['bench']
        arch = config['arch']
        mode = config['mode']
        job_file_base = config['job_file_base']
        bench_results_path = f"{job_file_base}.csv"
        bench_times_path = f"{job_file_base}.time.txt"

        try:
            regular_data = np.fromiter(ReadValues(bench_results_path), dtype=object, count=-1)
            avg_time = read_time(bench_times_path).mean()['time_var']
            data_row1 = {
                'bench': bench,
                'arch': arch,
                'mode': mode,
                'job_file_base': job_file_base,
                'bench_type': 'regular',
                'data': regular_data,
                'avg_time': avg_time
            }
            row_df = pd.DataFrame([data_row1])
            bench_results = pd.concat([bench_results, row_df], ignore_index=True)
        except Exception as inst:
            print(f"{bench}_{arch}_{mode} ops stats compilation error: {str(inst)}")

    for index, config in configs.iterrows():
        bench = config['bench']
        arch = config['arch']
        mode = config['mode']
        job_file_base = config['job_file_base']
        stats_job_file_base = config['stats_job_file_base']
        ops_stats_path = f"{job_file_base}.mix.txt"
        print(ops_stats_path)
        ops_placeholder = 0
        try:
            ops_stats = pd.read_csv(ops_stats_path, sep=" ", header=None).transpose()
            header = ops_stats.iloc[0]
            ops_stats = ops_stats[1:]
            ops_stats.columns=header
        except:
            print(f"{bench}_{arch}_{mode} ops stats compilation error")
            ops_stats = pd.DataFrame()
            ops_stats = ops_stats.concat([ops_stats, pd.DataFrame([{}])])
            ops_placeholder = np.nan

        precise_data = highest_precision[(highest_precision['bench'] == bench)].iloc[0]['data']
        approx_data = bench_results[
            (bench_results['bench'] == bench) &
            (bench_results['arch'] == arch) &
            (bench_results['mode'] == mode)
            ].iloc[0]['data']
        err_metrics = ComputeDifference(precise_data, approx_data)
        float_time = bench_results[(bench_results['arch'] == arch) &
                                   (bench_results['bench'] == bench) & (bench_results['mode'] == 'float')].iloc[0]['avg_time']
        mode_time = bench_results[(bench_results['arch'] == arch) &
                                   (bench_results['bench'] == bench) & (bench_results['mode'] == mode)].iloc[0]['avg_time']
        speedup = {
            'avg_time': mode_time,
            'speedup': float_time / mode_time
        }

        # print(ops_stats)
        stats_row = {
            'bench': bench,
            'arch': arch,
            'mode': mode,
            'job_file_base': job_file_base,
            'stats_job_file_base': stats_job_file_base,
            'MathOp': ops_stats.iloc[0].get("MathOp", ops_placeholder),
            'IntegerOp': ops_stats.iloc[0].get("IntegerOp", ops_placeholder),
            'Integer32Op': ops_stats.iloc[0].get("Integer32Op", ops_placeholder),
            'Integer64Op': ops_stats.iloc[0].get("Integer64Op", ops_placeholder),
            'FloatingPointOp': ops_stats.iloc[0].get("FloatingPointOp", ops_placeholder),
            'FloatSingleOp': ops_stats.iloc[0].get("FloatSingleOp", ops_placeholder),
            'FloatDoubleOp': ops_stats.iloc[0].get("FloatDoubleOp", ops_placeholder),
            'FloatMulDivOp': ops_stats.iloc[0].get("FloatMulDivOp", ops_placeholder),
            'FloatMulDivSingleOp': ops_stats.iloc[0].get("FloatMulDivSingleOp", ops_placeholder),
            'FloatMulDivDoubleOp': ops_stats.iloc[0].get("FloatMulDivDoubleOp", ops_placeholder),
            'smul.fix.i32': ops_stats.iloc[0].get("call(llvm.smul.fix.i32)", ops_placeholder),
            'sdiv.fix.i32': ops_stats.iloc[0].get("call(llvm.sdiv.fix.i32)", ops_placeholder),
            'add': ops_stats.iloc[0].get("add", ops_placeholder),
            'sub': ops_stats.iloc[0].get("sub", ops_placeholder),
            'div': ops_stats.iloc[0].get("div", ops_placeholder),
            'mul': ops_stats.iloc[0].get("mul", ops_placeholder),
            'CastOp': ops_stats.iloc[0].get("CastOp", ops_placeholder),
            'Shift': ops_stats.iloc[0].get("Shift", ops_placeholder),
            'fdiv': ops_stats.iloc[0].get("fdiv", ops_placeholder),
            'fmul': ops_stats.iloc[0].get("fmul", ops_placeholder),
            'fadd': ops_stats.iloc[0].get("fadd", ops_placeholder),
            'fsub': ops_stats.iloc[0].get("fsub", ops_placeholder),
        }
        stats_row.update(err_metrics)
        stats_row.update(speedup)
        row_df = pd.DataFrame([stats_row])
        stats_summary = pd.concat([stats_summary, row_df])

    mode_order = ["float", "fixed", "dynamic"]
    mode_type = CategoricalDtype(categories=mode_order, ordered=True)
    stats_summary['mode'] = stats_summary['mode'].astype(mode_type)
    stats_summary = stats_summary.sort_values(['bench', 'arch', 'mode'], ascending=False)
    print(stats_summary)
    stats_summary.to_csv(f'{results_path}/stats_summary.csv', index = None, header=True)

    plot_mode(results_path, stats_summary)
    plot_error(results_path, stats_summary)
    plot_error_bar(results_path, stats_summary)
    # plt.show()


def plot_error(results_path, stats_summary_full):
    file_base = f'{results_path}/errors'

    fig, axd = plt.subplot_mosaic([['pc_error', 'embedded_error']],
                                  constrained_layout=True)
    fig.suptitle('log2(relative error)')
    fig.set_size_inches(16, 8, forward=True)

    plot_error_single(stats_summary_full, 'PC', axd['pc_error'])
    plot_error_single(stats_summary_full, 'EMBEDDED', axd['embedded_error'])

    fig.savefig(f'{file_base}_relative.png', dpi=fig.dpi)

def plot_error_single(stats_summary_full, arch, ax):
    data_slice = stats_summary_full[(stats_summary_full['arch'] == arch)]
    data_err = data_slice.pivot(index='bench', columns='mode', values='e_perc_order')
    sns.heatmap(data_err, annot=True, linewidths=.5, ax=ax, cmap='coolwarm', vmin=-25, vmax=25)
    ax.title.set_text(f"{arch} error")

def plot_error_bar(results_path, stats_summary_full):
    file_base = f'{results_path}/stats'

    fig, axd = plt.subplot_mosaic([['pc_error', 'pc_speedup'], ['embedded_error', 'embedded_speedup']],
                                  constrained_layout=True)
    fig.suptitle(f'Relative error')
    fig.set_size_inches(14, 14, forward=True)
    plot_error_bar_single(stats_summary_full, 'PC', axd['pc_error'])
    plot_error_bar_single(stats_summary_full, 'EMBEDDED', axd['embedded_error'])
    plot_speedup_bar_single(stats_summary_full, 'PC', axd['pc_speedup'])
    plot_speedup_bar_single(stats_summary_full, 'EMBEDDED', axd['embedded_speedup'])
    fig.savefig(f'{file_base}_bar.png', dpi=fig.dpi)
    # fig.savefig(f'{file_base}.pdf', dpi=fig.dpi)

def plot_error_bar_single(stats_summary_full, arch, axd):
    pc_error = stats_summary_full[(stats_summary_full['arch'] == arch)]
    df = pc_error.pivot_table(index=['bench'], columns='mode', values='rel_err').reset_index()
    df = df.sort_values('bench', ascending=False)
    df.plot.barh(x='bench', y=['fixed', 'dynamic'],
                 ax=axd, log=True)
    axd.axvline(x=10**-1,linewidth=2, color='r')
    axd.set_axisbelow(True)
    axd.grid(color='gray', linestyle='dashed')
    axd.legend(loc='lower right')
    axd.set_xlabel('relative error')
    axd.set_ylabel('')
    axd.title.set_text(f"{arch} error")

def plot_speedup_bar_single(stats_summary_full, arch, axd):
    pc_error = stats_summary_full[(stats_summary_full['arch'] == arch)]
    df = pc_error.pivot_table(index=['bench'], columns='mode', values='speedup').reset_index()
    df = df.sort_values('bench', ascending=False)
    df.plot.barh(x='bench', y=['fixed', 'dynamic'],
                 ax=axd)
    axd.axvline(x=1,linewidth=2, color='r')
    axd.set_axisbelow(True)
    axd.grid(color='gray', linestyle='dashed')
    axd.legend(loc='lower right')
    axd.set_xlabel('speedup')
    axd.set_ylabel('')
    axd.title.set_text(f"{arch} speedup")

def plot_mode(results_path, stats_summary_full):
    file_base = f'{results_path}/modes'

    # print((stats_summary_full['mode'] == mode) & (stats_summary_full['scale'] == 1))
    # stats_summary = stats_summary_full
    stats_summary = stats_summary_full[
        (stats_summary_full['arch'] == 'EMBEDDED')
        ]

    fig, ax = plt.subplots(2, 2, constrained_layout=True)
    fig.set_size_inches(18, 12, forward=True)
    smul = stats_summary.pivot(index='bench', columns='mode', values='mul')
    sns.heatmap(smul, annot=True, linewidths=.5, ax=ax[0, 0], cmap='Blues')
    ax[0, 0].title.set_text('mul')
    sdiv = stats_summary.pivot(index='bench', columns='mode', values='div')
    sns.heatmap(sdiv, annot=True, linewidths=.5, ax=ax[1, 0], cmap='Blues')
    ax[1, 0].title.set_text('div')
    fmul = stats_summary.pivot(index='bench', columns='mode', values='fmul')
    sns.heatmap(fmul, annot=True, linewidths=.5, ax=ax[0, 1], cmap='Blues')
    ax[0, 1].title.set_text('fmul')
    fdiv = stats_summary.pivot(index='bench', columns='mode', values='fdiv')
    sns.heatmap(fdiv, annot=True, linewidths=.5, ax=ax[1, 1], cmap='Blues')
    ax[1, 1].title.set_text('fdiv')
    fig.savefig(f'{file_base}_mul_div.png', dpi=fig.dpi)

    fig2, ax2 = plt.subplots(2, 2, constrained_layout=True)
    fig2.set_size_inches(18, 12, forward=True)
    intop = stats_summary.pivot(index='bench', columns='mode', values='IntegerOp')
    sns.heatmap(intop, annot=True, linewidths=.5, ax=ax2[0, 0], cmap='Blues')
    ax2[0, 0].title.set_text('IntegerOp')
    floatop = stats_summary.pivot(index='bench', columns='mode', values='FloatingPointOp')
    sns.heatmap(floatop, annot=True, linewidths=.5, ax=ax2[1, 0], cmap='Blues')
    ax2[1, 0].title.set_text('FloatingPointOp')
    mathop = stats_summary.pivot(index='bench', columns='mode', values='MathOp')
    sns.heatmap(mathop, annot=True, linewidths=.5, ax=ax2[0, 1], cmap='Blues')
    ax2[0, 1].title.set_text('MathOp')
    shiftop = stats_summary.pivot(index='bench', columns='mode', values='Shift')
    sns.heatmap(shiftop, annot=True, linewidths=.5, ax=ax2[1, 1], cmap='Blues')
    ax2[1, 1].title.set_text('Shift')
    fig2.savefig(f'{file_base}_int_float.png', dpi=fig2.dpi)

    fig3, ax3 = plt.subplots(2, 2, constrained_layout=True)
    fig3.set_size_inches(18, 12, forward=True)
    add = stats_summary.pivot(index='bench', columns='mode', values='add')
    sns.heatmap(add, annot=True, linewidths=.5, ax=ax3[0, 0], cmap='Blues')
    ax3[0, 0].title.set_text('add')
    fadd = stats_summary.pivot(index='bench', columns='mode', values='fadd')
    sns.heatmap(fadd, annot=True, linewidths=.5, ax=ax3[0, 1], cmap='Blues')
    ax3[0, 1].title.set_text('fadd')
    sub = stats_summary.pivot(index='bench', columns='mode', values='sub')
    sns.heatmap(sub, annot=True, linewidths=.5, ax=ax3[1, 0], cmap='Blues')
    ax3[1, 0].title.set_text('sub')
    fsub = stats_summary.pivot(index='bench', columns='mode', values='fsub')
    sns.heatmap(fsub, annot=True, linewidths=.5, ax=ax3[1, 1], cmap='Blues')
    ax3[1, 1].title.set_text('fsub')
    fig3.savefig(f'{file_base}_add_sub.png', dpi=fig3.dpi)


def read_time(data_path):
    df = pd.read_csv(data_path, sep='\s+', names=['marker', 'time_var', 'unit_id'], dtype={'time_var': np.float64})
    df = df.drop(columns=['marker', 'unit_id'])
    return df
def ReadValues(filename):
    with open(filename, 'r') as f:
        l = f.readline()
        while l != '':
            for v in l.strip().split():
                if v != '' and v != '==BEGIN_DUMP_ARRAYS==' and v != '==END_DUMP_ARRAYS==':
                    yield v
            l = f.readline()
def ComputeDifference(taffo_values, float_values):
    abs_err = []
    rel_err = []
    max_abs = 0
    max_rel = 0
    for float_value,taffo_value in zip(float_values, taffo_values):
        float_value = float(float_value)
        taffo_value = float(taffo_value)
        if (float_value == 0 or taffo_value == 0):
            continue
        tmp = abs(taffo_value-float_value)
        abs_err.append(tmp)
        rel_err.append(tmp/abs(float_value))
        max_abs = max(max_abs, tmp)
        max_rel = max(max_rel, tmp/abs(float_value))
    abs_err = np.mean(abs_err)
    rel_err = np.mean(rel_err)
    e_perc_order = np.ceil(np.log2(float(rel_err)))
    if np.isinf(e_perc_order):
        e_perc_order = -25 # min error
    e_abs_order = np.ceil(np.log2(float(abs_err)))

    return {
        'e_perc_order': e_perc_order,
        'e_abs_order': e_abs_order,
        'output_size': float_values.size,
        "rel_err": rel_err,
        "max_rel": max_rel,
        "abs_err": abs_err,
        "max_abs": max_abs
    }

if __name__ == "__main__":
    main(sys.argv[1:])