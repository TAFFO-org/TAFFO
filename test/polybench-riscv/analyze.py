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
            fz_stats_path = join(stats_path, scale, bench, f"{bench}_float_size.csv")
            trace_path = join(stats_path, scale, bench, f"{bench}.instrumented.trace")
            print(var_stats_path)
            print(ops_stats_path)
            print(fz_stats_path)
            var_stats = pd.read_csv(var_stats_path)
            # print(var_stats)
            ops_placeholder = 0
            try:
                ops_stats = pd.read_csv(ops_stats_path, sep=" ", header=None).transpose()
                header = ops_stats.iloc[0]
                ops_stats = ops_stats[1:]
                ops_stats.columns=header
            except:
                print(f"{bench}_{scale} ops stats compilation error")
                ops_stats = pd.DataFrame()
                ops_stats = ops_stats.append({}, ignore_index=True)
                ops_placeholder = np.nan

            fz_placeholder = 0
            try:
                fz_stats = pd.read_csv(fz_stats_path, sep=",")
            except:
                print(f"{bench}_{scale} float size compilation error")
                fz_stats = pd.read_csv(StringIO("op_type,op1_range_set,op2_range_set,op0_range_min,op0_range_max,op1_range_min,op1_range_max,op0_range_normal,op1_range_normal,op0_exponent_min,op0_exponent_max,op1_exponent_min,op1_exponent_max,max_exponent_diff,"), sep=",")
                fz_placeholder = np.nan

            # print(fz_stats)

            fz_stats_valid = fz_stats.loc[(
                (fz_stats['op1_range_set'] > 0) &
                (fz_stats['op2_range_set'] > 0) &
                (fz_stats['op0_range_normal'] > 0) &
                (fz_stats['op1_range_normal'] > 0)
            )]
            print(fz_stats_valid)

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
                'scale': int(scale),
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
                'total_float': len(fz_stats),
                'float_finite_ranges': len(fz_stats_valid),
                'min_exp_diff': min(fz_stats_valid['max_exponent_diff']),
                'max_exp_diff': max(fz_stats_valid['max_exponent_diff']),
                'exp_diff_lt7': len(fz_stats_valid[fz_stats_valid['max_exponent_diff'] < 7]),
                'exp_diff_lt10': len(fz_stats_valid[fz_stats_valid['max_exponent_diff'] < 10]),
                'exp_diff_lt15': len(fz_stats_valid[fz_stats_valid['max_exponent_diff'] < 15]),
                'exp_diff_lt23': len(fz_stats_valid[fz_stats_valid['max_exponent_diff'] < 23]),
                'exp_diff_lt7_rel': safe_div(len(fz_stats_valid[fz_stats_valid['max_exponent_diff'] < 7]), len(fz_stats_valid)),
                'exp_diff_lt10_rel': safe_div(len(fz_stats_valid[fz_stats_valid['max_exponent_diff'] < 10]), len(fz_stats_valid)),
                'exp_diff_lt15_rel': safe_div(len(fz_stats_valid[fz_stats_valid['max_exponent_diff'] < 15]), len(fz_stats_valid)),
                'exp_diff_lt23_rel': safe_div(len(fz_stats_valid[fz_stats_valid['max_exponent_diff'] < 23]), len(fz_stats_valid)),
                'err_float16': err_float16,
                'err_float19': err_float19,
                'err_float24': err_float24,
                'err_float32': err_float32,
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
    fig.set_size_inches(18, 12, forward=True)

    smul = stats_summary.pivot(index='bench', columns='scale', values='smul.fix.i32')
    sns.heatmap(smul, annot=True, linewidths=.5, ax=ax[0,0], cmap='Blues')
    ax[0,0].title.set_text('smul.fix.i32')

    sdiv = stats_summary.pivot(index='bench', columns='scale', values='sdiv.fix.i32')
    sns.heatmap(sdiv, annot=True, linewidths=.5, ax=ax[1,0], cmap='Blues')
    ax[1,0].title.set_text('sdiv.fix.i32')

    fmul = stats_summary.pivot(index='bench', columns='scale', values='fmul')
    sns.heatmap(fmul, annot=True, linewidths=.5, ax=ax[0,1], cmap='Blues')
    ax[0,1].title.set_text('fmul')

    fdiv = stats_summary.pivot(index='bench', columns='scale', values='fdiv')
    sns.heatmap(fdiv, annot=True, linewidths=.5, ax=ax[1,1], cmap='Blues')
    ax[1,1].title.set_text('fdiv')

    fig.savefig(f'{stats_path}/mul_div.png', dpi=fig.dpi)


    fig2, ax2 = plt.subplots(2, 2, constrained_layout=True)
    fig2.set_size_inches(18, 12, forward=True)

    intop = stats_summary.pivot(index='bench', columns='scale', values='IntegerOp')
    sns.heatmap(intop, annot=True, linewidths=.5, ax=ax2[0,0], cmap='Blues')
    ax2[0,0].title.set_text('IntegerOp')

    floatop = stats_summary.pivot(index='bench', columns='scale', values='FloatingPointOp')
    sns.heatmap(floatop, annot=True, linewidths=.5, ax=ax2[1,0], cmap='Blues')
    ax2[1,0].title.set_text('FloatingPointOp')

    mathop = stats_summary.pivot(index='bench', columns='scale', values='MathOp')
    sns.heatmap(mathop, annot=True, linewidths=.5, ax=ax2[0,1], cmap='Blues')
    ax2[0,1].title.set_text('MathOp')

    shiftop = stats_summary.pivot(index='bench', columns='scale', values='Shift')
    sns.heatmap(shiftop, annot=True, linewidths=.5, ax=ax2[1,1], cmap='Blues')
    ax2[1,1].title.set_text('Shift')

    fig2.savefig(f'{stats_path}/int_float.png', dpi=fig2.dpi)


    fig3, ax3 = plt.subplots(2, 2, constrained_layout=True)
    fig3.set_size_inches(18, 12, forward=True)

    add = stats_summary.pivot(index='bench', columns='scale', values='add')
    sns.heatmap(add, annot=True, linewidths=.5, ax=ax3[0,0], cmap='Blues')
    ax3[0,0].title.set_text('add')

    fadd = stats_summary.pivot(index='bench', columns='scale', values='fadd')
    sns.heatmap(fadd, annot=True, linewidths=.5, ax=ax3[0,1], cmap='Blues')
    ax3[0,1].title.set_text('fadd')

    sub = stats_summary.pivot(index='bench', columns='scale', values='sub')
    sns.heatmap(sub, annot=True, linewidths=.5, ax=ax3[1,0], cmap='Blues')
    ax3[1,0].title.set_text('sub')

    fsub = stats_summary.pivot(index='bench', columns='scale', values='fsub')
    sns.heatmap(fsub, annot=True, linewidths=.5, ax=ax3[1,1], cmap='Blues')
    ax3[1,1].title.set_text('fsub')

    fig3.savefig(f'{stats_path}/add_sub.png', dpi=fig3.dpi)

    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])