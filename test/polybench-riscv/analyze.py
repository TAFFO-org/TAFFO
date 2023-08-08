#!/usr/bin/python3

import sys
import pandas as pd
import os
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
                          names=["bench", "input_size", "scaling", "mode", "mantissa", "job_file_base", "stats_job_file_base"])
    double_configs = pd.read_csv(double_configs_file_path, sep=",",
                          names=["bench", "input_size", "scaling", "job_file_base"])
    print(configs)
    stats_summary = pd.DataFrame()
    bench_results = pd.DataFrame()
    highest_precision = pd.DataFrame()
    for index, config in double_configs.iterrows():
        bench = config['bench']
        input_size = config['input_size']
        scaling = config['scaling']
        job_file_base = config['job_file_base']
        bench_results_path = f"{job_file_base}.csv"

        try:
            regular_data = np.fromiter(ReadValues(bench_results_path), dtype=object, count=-1)
            data_row1 = {
                'bench': bench,
                'input_size': input_size,
                'scale': int(scaling),
                'job_file_base': job_file_base,
                'bench_type': 'double',
                'data': regular_data
            }
            row_df = pd.DataFrame([data_row1])
            highest_precision = pd.concat([highest_precision, row_df], ignore_index=True)
        except Exception as inst:
            print(f"{bench}_{scaling} double ops stats compilation error: {str(inst)}")

    for index, config in configs.iterrows():
        bench = config['bench']
        input_size = config['input_size']
        scaling = config['scaling']
        mode = config['mode']
        mantissa = config['mantissa']
        job_file_base = config['job_file_base']
        bench_results_path = f"{job_file_base}.csv"
        bench_lamp_results_path = f"{job_file_base}.lamp.csv"

        try:
            regular_data = np.fromiter(ReadValues(bench_results_path), dtype=object, count=-1)
            lamp_data = np.fromiter(ReadValues(bench_lamp_results_path), dtype=object, count=-1)
            data_row1 = {
                'bench': bench,
                'input_size': input_size,
                'scale': int(scaling),
                'mode': mode,
                'mantissa': int(mantissa),
                'job_file_base': job_file_base,
                'bench_type': 'regular',
                'data': regular_data
            }
            data_row2 = {
                'bench': bench,
                'input_size': input_size,
                'scale': int(scaling),
                'mode': mode,
                'mantissa': int(mantissa),
                'job_file_base': job_file_base,
                'bench_type': 'lamp',
                'data': lamp_data
            }
            row_df = pd.DataFrame([data_row1, data_row2])
            bench_results = pd.concat([bench_results, row_df], ignore_index=True)
        except Exception as inst:
            print(f"{bench}_{scaling} ops stats compilation error: {str(inst)}")

    for index, config in configs.iterrows():
        bench = config['bench']
        input_size = config['input_size']
        scaling = config['scaling']
        mode = config['mode']
        mantissa = config['mantissa']
        job_file_base = config['job_file_base']
        stats_job_file_base = config['stats_job_file_base']
        var_stats_path = f"{stats_job_file_base}.csv"
        ops_stats_path = f"{job_file_base}.mix.txt"
        trace_path = f"{stats_job_file_base}.instrumented.trace"
        print(var_stats_path)
        print(ops_stats_path)
        var_stats = pd.read_csv(var_stats_path)
        # print(var_stats)
        ops_placeholder = 0
        try:
            ops_stats = pd.read_csv(ops_stats_path, sep=" ", header=None).transpose()
            header = ops_stats.iloc[0]
            ops_stats = ops_stats[1:]
            ops_stats.columns=header
        except:
            print(f"{bench}_{scaling} ops stats compilation error")
            ops_stats = pd.DataFrame()
            ops_stats = pd.concat([ops_stats, pd.DataFrame([{}])])
            ops_placeholder = np.nan

        float_op_count = 0
        by_opcode_count = {
            'fmul': 0,
            'fdiv': 0,
            'fadd': 0,
            'fsub': 0,
        }
        float_op_abs_max = 0
        by_opcode_abs_max = {
            'fmul': 0,
            'fdiv': 0,
            'fadd': 0,
            'fsub': 0,
        }

        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                parts = line.split(" ")
                if len(parts) == 0 or parts[0] != "TAFFO_TRACE": continue
                value = float.fromhex(parts[2])
                opcode = parts[4]
                # print(opcode, value)
                if opcode == 'fmul' or opcode == 'fdiv' or opcode == 'fadd' or opcode == 'fsub':
                    float_op_count += 1
                    by_opcode_count[opcode] += 1
                    float_op_abs_max = max(float_op_abs_max, abs(value))
                    by_opcode_abs_max[opcode] = max(by_opcode_abs_max[opcode], abs(value))

        try:
            # print(highest_precision[(highest_precision['bench'] == bench)].iloc[0]['data'])
            precise_data = highest_precision[((highest_precision['bench'] == bench)) &
                                             (highest_precision['scale'] == int(scaling))].iloc[0]['data']
            approx_data = bench_results[
                (bench_results['bench'] == bench) &
                (bench_results['input_size'] == input_size) &
                (bench_results['scale'] == int(scaling)) &
                (bench_results['mode'] == mode) &
                (bench_results['bench_type'] == 'lamp') &
                (bench_results['mantissa'] == int(mantissa))
            ].iloc[0]['data']
            err_metrics = ComputeDifference(precise_data, approx_data)

            logn = np.ceil(np.log2(float_op_count))
            # log_max_value = np.ceil(np.log2(float_op_abs_max))
            # err_float16 = logn - 7
            # err_float19 = logn - 10
            # err_float24 = logn - 15
            # err_float32 = logn - 23
            err_perc_predicted_order = logn - int(mantissa) + 1
            err_perc_predicted_order_adjusted = logn - int(mantissa) + 1 - np.floor(np.log2(err_metrics['output_size']))

            # print(ops_stats)
            stats_row = {
                'bench': bench,
                'input_size': input_size,
                'scale': int(scaling),
                'mode': mode,
                'mantissa': int(mantissa),
                'float_type': (f"float{int(mantissa) + 8}" if (int(mantissa) != 8) else "bfloat16"),
                'job_file_base': job_file_base,
                'stats_job_file_base': stats_job_file_base,
                'var_min': var_stats['var_min'].min(),
                'var_max': var_stats['var_max'].max(),
                'var_isnan': var_stats['var_isnan'].max(),
                'var_isinf': var_stats['var_isinf'].max(),
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
                'CastOp': ops_stats.iloc[0].get("CastOp", ops_placeholder),
                'Shift': ops_stats.iloc[0].get("Shift", ops_placeholder),
                'fdiv': ops_stats.iloc[0].get("fdiv", ops_placeholder),
                'fmul': ops_stats.iloc[0].get("fmul", ops_placeholder),
                'fadd': ops_stats.iloc[0].get("fadd", ops_placeholder),
                'fsub': ops_stats.iloc[0].get("fsub", ops_placeholder),
                'err_predicted_perc_order': err_perc_predicted_order,
                'err_predicted_perc_order_adjusted': err_perc_predicted_order_adjusted,
                'float_op_count': float_op_count
                # 'err_float16': err_float16,
                # 'err_float19': err_float19,
                # 'err_float24': err_float24,
                # 'err_float32': err_float32,
            }
            stats_row.update(err_metrics)
            row_df = pd.DataFrame([stats_row])
            stats_summary = pd.concat([stats_summary, row_df])
        except:
            print("could not handle row")

    mode_order = ["float", "fixed", "mixed"]
    mode_type = CategoricalDtype(categories=mode_order, ordered=True)
    stats_summary['mode'] = stats_summary['mode'].astype(mode_type)
    stats_summary = stats_summary.sort_values(['bench', 'input_size', 'scale', 'mode', 'mantissa'], ascending=False)
    stats_summary["bench_scale"] = stats_summary['bench'].astype(str) +"-"+ stats_summary["scale"].astype(str)
    print(stats_summary)
    stats_summary.to_csv(f'{results_path}/stats_summary.csv', index = None, header=True)

    scales = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for s in scales:
        stats_summary1 = stats_summary[stats_summary['scale'].eq(s)]
        plot_mode(f"{results_path}/{s}", stats_summary1)
        plot_error(f"{results_path}/{s}", stats_summary1)
        plt.close('all')


def plot_error(results_path, stats_summary_full):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    file_base = f'{results_path}/errors'

    stats_summary_float = stats_summary_full[(stats_summary_full['mode'] == 'float')]
    stats_summary_fixed = stats_summary_full[(stats_summary_full['mode'] == 'fixed')]
    stats_summary_mixed = stats_summary_full[(stats_summary_full['mode'] == 'mixed')]

    fig, axd = plt.subplot_mosaic([['float', 'fixed', 'mixed'],['predicted', 'predicted_adjusted', 'empty']], constrained_layout=True)
    fig.suptitle('Calculated and predicted log2(relative error)')
    fig.set_size_inches(22, 12, forward=True)
    float_err = stats_summary_float.pivot(index='bench', columns='float_type', values='e_perc_order')
    sns.heatmap(float_err, annot=True, linewidths=.5, ax=axd['float'], cmap='coolwarm', vmin=-25, vmax=25)
    axd['float'].title.set_text('float')
    fixed_err = stats_summary_fixed.pivot(index='bench', columns='float_type', values='e_perc_order')
    sns.heatmap(fixed_err, annot=True, linewidths=.5, ax=axd['fixed'], cmap='coolwarm', vmin=-25, vmax=25)
    axd['fixed'].title.set_text('fixed')
    mixed_err = stats_summary_mixed.pivot(index='bench', columns='float_type', values='e_perc_order')
    sns.heatmap(mixed_err, annot=True, linewidths=.5, ax=axd['mixed'], cmap='coolwarm', vmin=-25, vmax=25)
    axd['mixed'].title.set_text('mixed')
    predicted_err = stats_summary_float.pivot(index='bench', columns='float_type', values='err_predicted_perc_order')
    sns.heatmap(predicted_err, annot=True, linewidths=.5, ax=axd['predicted'], cmap='coolwarm', vmin=-25, vmax=25)
    axd['predicted'].title.set_text('predicted')
    predicted_err = stats_summary_float.pivot(index='bench', columns='float_type', values='err_predicted_perc_order_adjusted')
    sns.heatmap(predicted_err, annot=True, linewidths=.5, ax=axd['predicted_adjusted'], cmap='coolwarm', vmin=-25, vmax=25)
    axd['predicted_adjusted'].title.set_text('predicted adjusted to output size')
    fig.savefig(f'{file_base}_relative.png', dpi=fig.dpi)

def plot_mode(results_path, stats_summary_full):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    file_base = f'{results_path}/modes'

    # print((stats_summary_full['mode'] == mode) & (stats_summary_full['scale'] == 1))
    stats_summary = stats_summary_full[
        (stats_summary_full['mantissa'] == 24)
    ]

    fig, ax = plt.subplots(2, 2, constrained_layout=True)
    fig.set_size_inches(18, 12, forward=True)
    smul = stats_summary.pivot(index='bench', columns='mode', values='smul.fix.i32')
    sns.heatmap(smul, annot=True, linewidths=.5, ax=ax[0, 0], cmap='Blues')
    ax[0, 0].title.set_text('smul.fix.i32')
    sdiv = stats_summary.pivot(index='bench', columns='mode', values='sdiv.fix.i32')
    sns.heatmap(sdiv, annot=True, linewidths=.5, ax=ax[1, 0], cmap='Blues')
    ax[1, 0].title.set_text('sdiv.fix.i32')
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


def ReadValues(filename):
    with open(filename, 'r') as f:
        l = f.readline()
        while l != '':
            for v in l.strip().split():
                if v != '':
                    yield v
            l = f.readline()
def ComputeDifference(fix_data, flt_data):
    n = 0
    accerr = Decimal(0)
    accval = Decimal(0)
    fix_nofl = 0
    flo_nofl = 0

    try:
        thres_ofl_cp = Decimal('0.01')

        for svfix, svflo in zip(fix_data, flt_data):
            vfix, vflo = Decimal(svfix), Decimal(svflo)

            if not vfix.is_finite():
                fix_nofl += 1
            elif not vflo.is_finite():
                flo_nofl += 1
                fix_nofl += 1
            elif ((vflo + vfix).copy_abs() - (vflo.copy_abs() + vfix.copy_abs())) > thres_ofl_cp:
                fix_nofl += 1
            else:
                n += 1
                accerr += (vflo - vfix).copy_abs()
                accval += vflo

        e_perc = (accerr / accval.copy_abs() * 100) if accval != 0 and n > 0 else -1
        e_abs = (accerr / n) if n > 0 else -1

        e_perc_order = np.ceil(np.log2(float(e_perc)))
        if np.isinf(e_perc_order):
            e_perc_order = -25 # min error
        e_abs_order = np.ceil(np.log2(float(e_abs)))
    except Exception as inst:
        print(f"exception when computing error: {str(inst)}")
        return {
            'e_perc_order': np.nan,
            'e_abs_order': np.nan,
            'fix_nofl': 0,
            'flo_nofl': 0,
            'e_perc': np.nan,
            'e_abs': np.nan,
            'output_size': 0,
        }


    return {
        'e_perc_order': e_perc_order,
        'e_abs_order': e_abs_order,
        'fix_nofl': fix_nofl,
        'flo_nofl': flo_nofl,
        'e_perc': e_perc,
        'e_abs': e_abs,
        'output_size': n,
    }

if __name__ == "__main__":
    main(sys.argv[1:])