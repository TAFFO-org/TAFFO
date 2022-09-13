#!/usr/bin/python3

import sys
import pandas as pd
from os import listdir
from os.path import isfile, join, isdir
import matplotlib.pyplot as plt

pd.options.display.max_columns = None

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
    stats_summary.plot(x='bench_scale',
                        y=['smul.fix.i32', 'sdiv.fix.i32', 'fdiv', 'fmul'],
                        kind='barh',
                        stacked=True
                       )

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])