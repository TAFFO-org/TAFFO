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

pd.options.display.max_columns = None
# plt.rcParams.update({'font.size': 8})

def safe_div(x,y):
    if y == 0:
        return 0
    return x / y
def main(argv):
    build_dir = argv[0]
    configs_file_path = f"{build_dir}/configurations.csv"
    results_path = f"{build_dir}/summary"
    configs = pd.read_csv(configs_file_path, sep=",",
                          names=["bench", "input_size", "scaling", "mode", "mantissa", "job_file_base", "stats_job_file_base"])
    print(configs)
    stats_summary = pd.DataFrame()
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
            ops_stats = ops_stats.concat([ops_stats, pd.DataFrame([{}])])
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


        logn = math.ceil(math.log2(float_op_count))
        log_max_value = math.ceil(math.log2(float_op_abs_max))
        err_float16 = logn - 7
        err_float19 = logn - 10
        err_float24 = logn - 15
        err_float32 = logn - 23

        # print(ops_stats)
        stats_row = {
            'bench': bench,
            'input_size': input_size,
            'scale': int(scaling),
            'mode': mode,
            'mantissa': int(mantissa),
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
            'err_float16': err_float16,
            'err_float19': err_float19,
            'err_float24': err_float24,
            'err_float32': err_float32,
        }
        row_df = pd.DataFrame([stats_row])
        stats_summary = pd.concat([stats_summary, row_df])

    mode_order = ["float", "fixed", "mixed"]
    mode_type = CategoricalDtype(categories=mode_order, ordered=True)
    stats_summary['mode'] = stats_summary['mode'].astype(mode_type)
    stats_summary = stats_summary.sort_values(['bench', 'input_size', 'scale', 'mode', 'mantissa'], ascending=False)
    stats_summary["bench_scale"] = stats_summary['bench'].astype(str) +"-"+ stats_summary["scale"].astype(str)
    print(stats_summary)
    stats_summary.to_csv(f'{results_path}/stats_summary.csv', index = None, header=True)

    plot_mode(results_path, stats_summary)
    plt.show()


def plot_mode(results_path, stats_summary_full):
    file_base = f'{results_path}/modes'

    # print((stats_summary_full['mode'] == mode) & (stats_summary_full['scale'] == 1))
    stats_summary = stats_summary_full[
        (stats_summary_full['mantissa'] == 24) &
        (stats_summary_full['scale'] == 1)
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


if __name__ == "__main__":
    main(sys.argv[1:])