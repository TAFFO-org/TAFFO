#!/usr/bin/env python

import argparse
import shutil
from pathlib import Path
import subprocess
import re
from xxsubtype import bench

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import sys
import gmpy2
from gmpy2 import mpfr, trunc, log2
import platform
import scipy as sc
import warnings
from colorama import init as colorama_init, Fore, Style
import concurrent.futures
import os

colorama_init(autoreset=True)

gmpy2.get_context().precision = 100

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

CLANG = subprocess.check_output('taffo -print-clang', shell=True, encoding='utf8').strip()

ACTION_PAD = 0  # will compute padding once we know all test names

def bold(s: str):
    print(f"{Style.BRIGHT}{s}{Style.RESET_ALL}", flush=True)

def print_tables(table):
    df = pd.DataFrame(table)
    print(
        df.to_string(
            columns=[
                "name", "speedup", "compile_time",
                "rel_err", "abs_err", "pvalue"
            ],
            formatters={
                "speedup": lambda x: f"{x:.2f}x",
                "compile_time": lambda x: f"{x:.2f}s",
                "rel_err": "{:.2%}".format,
                "abs_err": "{:.2e}".format,
                "pvalue": "{:.2e}".format
            }
        ),
        flush=True
    )

def generatedata(path: Path):
    subprocess.run(
        "./datagenerator.py > data.h",
        cwd=path.as_posix(),
        shell=True
    )

def compile(path: Path):
    """
    Compile both the float and taffo versions of the benchmark in `path`.
    Returns a tuple (path, ok:bool, log:str).
    """
    logs = []
    bench_src = None
    for ext in [".c", ".cpp", ".ll"]:
        if (path/f"{path.name}{ext}").exists():
            bench_src = path / f"{path.name}{ext}"
            break
    if not bench_src:
        logs.append(f"Missing source for {path.name}\n")
        return "".join(logs)

    global common_args
    common_args += f" -I{path.absolute()}"

    # float build
    bench_f = f"{path.name}-float"
    label_f = f"Compiling: {bench_f}".ljust(ACTION_PAD)
    clang_cmd = 'clang -Wno-error=implicit-function-declaration' if platform.system()=="Darwin" else CLANG
    args_txt = (path / "args.txt").read_text().strip() if (path/"args.txt").exists() else ""
    cmd_f = f"{clang_cmd} {common_args} {args_txt} {bench_src} -o {bench_f} -lm"
    p_f = subprocess.run(cmd_f, cwd=str(path), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ok_f = (p_f.returncode == 0)
    if ok_f:
        (path/bench_f).chmod(0o755)
        logs.append(f"{label_f}{Fore.GREEN}{Style.BRIGHT}OKK!{Style.RESET_ALL}\n")
    else:
        logs.append(f"{label_f}{Fore.RED}{Style.BRIGHT}ERR!{Style.RESET_ALL}\n")
        logs.append(p_f.stderr.decode())

    # taffo build
    bench_t = f"{path.name}-taffo"
    label_t = f"Compiling: {bench_t}".ljust(ACTION_PAD)
    flag2 = f"{common_args} -time-profile-file {path}/{path.name}_taffo_time.csv"
    args2_txt = (path/"args_taffo.txt").read_text().strip() if (path/"args_taffo.txt").exists() else ""
    if debug:
        (path/"taffo_temp").mkdir(exist_ok=True)
        flag2 += " -debug -temp-dir ./taffo_temp"
        logf = path/f"{path.name}_taffo.log"
        po = open(logf, "w")
        p_t = subprocess.run(
            f"taffo {flag2} {args_txt} {args2_txt} {bench_src} -o {bench_t} -lm",
            cwd=str(path), shell=True, stdout=po, stderr=po
        )
        po.close()
    else:
        p_t = subprocess.run(
            f"taffo {flag2} {bench_src} -o {bench_t} -lm",
            cwd=str(path), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    ok_t = (p_t.returncode == 0)
    if ok_t:
        (path/bench_t).chmod(0o755)
        logs.append(f"{label_t}{Fore.GREEN}{Style.BRIGHT}OKK!{Style.RESET_ALL}\n")
    else:
        logs.append(f"{label_t}{Fore.RED}{Style.BRIGHT}ERR!{Style.RESET_ALL}\n")
        if not debug:
            logs.append(f"{Fore.RED}{p_t.stderr.decode()}{Style.RESET_ALL}")

    return path, ok_f and ok_t, "".join(logs)

def run(path: Path):
    """
    Runs the two binaries; returns True only if both exit zero.
    """
    bench = path.name
    f_exec = path/f"{bench}-float"
    t_exec = path/f"{bench}-taffo"
    inputs = sorted(path.glob("input*.txt"))
    ok_all = True

    if inputs:
        for inp in inputs:
            suffix = inp.stem[len("input"):]
            out_t = f"taffo-res{suffix}"
            out_f = f"float-res{suffix}"

            print(f"Running: {bench}{suffix}-float".ljust(ACTION_PAD), end="", flush=True)
            p1 = subprocess.run(f"./{bench}-float < {inp.name} > {out_f}", cwd=str(path), shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p1.returncode:
                ok_all = False
                print(f"{Fore.RED}{Style.BRIGHT}ERR!{Style.RESET_ALL}\n{Fore.RED}{p1.stderr.decode().strip()}{Style.RESET_ALL}", flush=True)
            else:
                print(f"{Fore.GREEN}{Style.BRIGHT}OKK!{Style.RESET_ALL}", flush=True)

            print(f"Running: {bench}{suffix}-taffo".ljust(ACTION_PAD), end="", flush=True)
            p2 = subprocess.run(f"./{bench}-taffo < {inp.name} > {out_t}", cwd=str(path), shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p2.returncode:
                ok_all = False
                print(f"{Fore.RED}{Style.BRIGHT}ERR!{Style.RESET_ALL}\n{Fore.RED}{p2.stderr.decode().strip()}{Style.RESET_ALL}", flush=True)
            else:
                print(f"{Fore.GREEN}{Style.BRIGHT}OKK!{Style.RESET_ALL}", flush=True)
    else:
        print(f"Running: {bench}-float".ljust(ACTION_PAD), end="", flush=True)
        p1 = subprocess.run(f"./{bench}-float > float-res", cwd=str(path), shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p1.returncode:
            ok_all = False
            print(f"{Fore.RED}{Style.BRIGHT}ERR!{Style.RESET_ALL}\n{Fore.RED}{p1.stderr.decode().strip()}{Style.RESET_ALL}", flush=True)
        else:
            print(f"{Fore.GREEN}{Style.BRIGHT}OKK!{Style.RESET_ALL}", flush=True)
        print(f"Running: {bench}-taffo".ljust(ACTION_PAD), end="", flush=True)
        p2 = subprocess.run(f"./{bench}-taffo > taffo-res", cwd=str(path), shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p2.returncode:
            ok_all = False
            print(f"{Fore.RED}{Style.BRIGHT}ERR!{Style.RESET_ALL}\n{Fore.RED}{p2.stderr.decode().strip()}{Style.RESET_ALL}", flush=True)
        else:
            print(f"{Fore.GREEN}{Style.BRIGHT}OKK!{Style.RESET_ALL}", flush=True)

    return ok_all

def retrieveFiles(path: Path):
    """
    Return a dict mapping each input-suffix ('' or '.1', '.2', etc.)
    to a (float_output, taffo_output) tuple of strings.
    """
    results = {}
    # look for any input*.txt
    inputs = sorted(path.glob("input*.txt"))
    if inputs:
        for inp in inputs:
            # compute suffix ('' or '.1', '.2', etc.)
            suffix = inp.stem[len("input"):]
            ffile = path / f"float-res{suffix}"
            tfile = path / f"taffo-res{suffix}"
            if not ffile.exists() or not tfile.exists():
                print(f"{Fore.RED}{Style.BRIGHT}ERR! missing results for suffix '{suffix}'{Style.RESET_ALL}")
                continue
            results[suffix] = (ffile.read_text(), tfile.read_text())
    else:
        # fallback to the un-suffixed files
        ffile = path / "float-res"
        tfile = path / "taffo-res"
        if not ffile.exists() or not tfile.exists():
            print(f"{Fore.RED}{Style.BRIGHT}ERR! missing float-res or taffo-res{Style.RESET_ALL}")
            return {}
        results[""] = (ffile.read_text(), tfile.read_text())
    return results

def retriveCompilationTime(path: Path):
    tf = path / f"{path.name}_taffo_time.csv"
    if not tf.exists():
        print(f"{Fore.RED}{Style.BRIGHT}ERR!{Style.RESET_ALL}", flush=True)
        return pd.DataFrame()
    return pd.read_csv(tf)

def reject_outliers(data):
    arr = np.array(data)
    m, sd = arr.mean(), arr.std()
    return arr[abs(arr - m) < 2 * sd]

def getTime(files):
    # Try integer cycle counts first
    cycles_f = re.findall(r"Cycles: (\d+)", files[0])
    cycles_t = re.findall(r"Cycles: (\d+)", files[1])

    if cycles_f and cycles_t:
        # Option 1: cycles as floats
        times_f = [float(x) for x in cycles_f]
        times_t = [float(x) for x in cycles_t]
    else:
        # Option 2: decimal times
        times_f = [float(x) for x in re.findall(r"Time: ([0-9]*\.?[0-9]+)", files[0])]
        times_t = [float(x) for x in re.findall(r"Time: ([0-9]*\.?[0-9]+)", files[1])]

    if len(times_f) > 1:
        # Reject outliers in both lists
        times_f = reject_outliers(times_f)
        times_t = reject_outliers(times_t)
    # Compute means (or NaN if empty)
    mean_f = np.mean(times_f) if len(times_f) else float('nan')
    mean_t = np.mean(times_t) if len(times_t) else float('nan')
    return mean_f, mean_t

def getData(files):
    fblock = re.search(r"Values Begin\n([\s\S]*?)\nValues End", files[0]).group(1)
    tblock = re.search(r"Values Begin\n([\s\S]*?)\nValues End", files[1]).group(1)

    abs_errs, rel_errs = [], []
    float_data, taffo_data = [], []
    max_abs = max_rel = 0

    for fv, tv in zip(fblock.splitlines(), tblock.splitlines()):
        # skip empty lines
        if not fv.strip() and not tv.strip():
            continue
        fv_f, tv_f = float(fv), float(tv)
        if fv_f == 0 or tv_f == 0:
            continue
        diff = abs(tv_f - fv_f)
        abs_errs.append(diff)
        rel = diff / abs(fv_f)
        rel_errs.append(rel)
        max_abs = max(max_abs, diff)
        max_rel = max(max_rel, rel)
        float_data.append(fv_f)
        taffo_data.append(tv_f)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pval = sc.stats.ks_2samp(float_data, taffo_data).pvalue

    return {
        "abs_err": np.mean(abs_errs),
        "max_abs": max_abs,
        "rel_err": np.mean(rel_errs),
        "max_rel": max_rel,
        "pvalue": pval,
    }

def ordereddiff(path: Path):
    files = retrieveFiles(path)
    fv = re.search(r"Values Begin\n([\s\S]*?)\nValues End", files[0]).group(1)
    tv = re.search(r"Values Begin\n([\s\S]*?)\nValues End", files[1]).group(1)
    errs = []
    for i, (fv_line, tv_line) in enumerate(zip(fv.splitlines(), tv.splitlines()), start=1):
        # skip empty lines
        if not fv_line.strip() and not tv_line.strip():
            continue
        fv_f, tv_f = float(fv_line), float(tv_line)
        if fv_f == 0 or tv_f == 0:
            continue
        errs.append((i, abs(tv_f - fv_f) / abs(fv_f)))
    errs.sort(key=lambda x: x[1], reverse=True)
    for line, err in errs:
        print(f"{line}, {err:.4%}")

def validate(path: Path, compute_speedup=True):
    compile_df = retriveCompilationTime(path)
    if compile_df.empty: return []
    compile_time = compile_df.iloc[0].taffo_end - compile_df.iloc[0].taffo_start
    results = []
    files_dict = retrieveFiles(path)
    for suffix, (f_txt, t_txt) in files_dict.items():
        ftime, ttime = getTime([f_txt, t_txt])
        datas = getData([f_txt, t_txt])
        speedup = (ftime/ttime) if compute_speedup else None
        results.append({
            "name": path.name + suffix,
            "speedup": speedup,
            "compile_time": compile_time,
            **datas
        })
    return results

def extract_compile_time(path: Path):
    df = retriveCompilationTime(path)
    row = df.iloc[0]
    return {
        "name": path.name,
        "total_time": row.taffo_end - row.taffo_start,
        "base_ll_time": row.init_start - row.taffo_start,
        "backend_time": row.taffo_end - row.backend_start,
        "taffo_init_time": row.vra_start - row.init_start,
        "vra_time": row.dta_start - row.vra_start,
        "dta_time": row.conversion_start - row.dta_start,
        "conversion_time": row.backend_start - row.conversion_start,
    }

def plot_compile_times(path: Path, table):
    table = pd.DataFrame(table)
    abs_plot_table = table.drop('total_time', axis=1)
    ax = abs_plot_table.plot.barh(x='name', stacked=True, title='TAFFO compilation stages time', figsize=(10, 8))
    fig = ax.get_figure()
    plt.xlabel("Seconds")
    plt.ylabel("Benchmark")
    fig.savefig(f'{path}/compile_time.png', dpi=fig.dpi, bbox_inches = 'tight')

    rel_plot_table = pd.DataFrame()
    rel_plot_table["name"] = (table["name"])
    rel_plot_table["base_ll_time_perc"] = (table["base_ll_time"] / table["total_time"])
    rel_plot_table["backend_time_perc"] = (table["backend_time"] / table["total_time"])
    rel_plot_table["taffo_init_time_perc"] = (table["taffo_init_time"] / table["total_time"])
    rel_plot_table["vra_time_perc"] = (table["vra_time"] / table["total_time"])
    rel_plot_table["dta_time_perc"] = (table["dta_time"] / table["total_time"])
    rel_plot_table["conversion_time_perc"] = (table["conversion_time"] / table["total_time"])
    # print(rel_plot_table)
    ax2 = rel_plot_table.plot.barh(x='name', stacked=True, title='TAFFO compilation stages time percentage', figsize=(10, 8))
    fig2 = ax2.get_figure()
    plt.xlabel("% Time")
    plt.ylabel("Benchmark")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.xaxis.set_major_formatter(PercentFormatter(1))
    for p in ax2.patches:
        h, w, x, y = p.get_height(), p.get_width(), p.get_x(), p.get_y()
        text = f'{w * 100:0.1f} %'
        ax2.annotate(text=text, xy=(x + w / 2, y + h / 2), ha='center', va='center', color='white', size=10)
    fig2.savefig(f'{path}/compile_time_rel.png', dpi=fig2.dpi, bbox_inches = 'tight')

    rel_taffo_plot_table = pd.DataFrame()
    rel_taffo_plot_table["name"] = (table["name"])
    rel_taffo_plot_table["base_ll_time_perc"] = (table["base_ll_time"] / (table["base_ll_time"] + table["backend_time"]))
    rel_taffo_plot_table["backend_time_perc"] = (table["backend_time"] / (table["base_ll_time"] + table["backend_time"]))
    rel_taffo_plot_table["taffo_init_time_perc"] = (table["taffo_init_time"] / (table["base_ll_time"] + table["backend_time"]))
    rel_taffo_plot_table["vra_time_perc"] = (table["vra_time"] / (table["base_ll_time"] + table["backend_time"]))
    rel_taffo_plot_table["dta_time_perc"] = (table["dta_time"] / (table["base_ll_time"] + table["backend_time"]))
    rel_taffo_plot_table["conversion_time_perc"] = (table["conversion_time"] / (table["base_ll_time"] + table["backend_time"]))
    ax3 = rel_taffo_plot_table.plot.barh(x='name', stacked=True, title='TAFFO compilation time (normalized to plain compilation)', figsize=(10, 8))
    fig3 = ax3.get_figure()
    plt.xlabel("% Time")
    plt.ylabel("Benchmark")
    plt.legend([
        "C to LLVM IR",
        "Backend",
        "TAFFO init",
        "VRA",
        "DTA",
        "Conversion",
    ],
    bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.xaxis.set_major_formatter(PercentFormatter(1))
    for p in ax3.patches:
        h, w, x, y = p.get_height(), p.get_width(), p.get_x(), p.get_y()
        text = f'{w * 100:0.1f} %'
        ax3.annotate(text=text, xy=(x + w / 2, y + h / 2), ha='center', va='center', color='white', size=10)
    fig3.savefig(f'{path}/compile_time_taffo_rel.png', dpi=fig3.dpi, bbox_inches = 'tight')
    print(rel_taffo_plot_table)

def clean(path: Path, suffix_to_keep: tuple):
    temp_dir = path/"taffo_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    for f in path.glob("*"):
        if f.suffix not in suffix_to_keep:
            if f.is_file():
                f.unlink()
        elif f.is_dir():
            clean(f, suffix_to_keep)
            try:
                f.rmdir()
            except OSError:
                pass

def extract_first_n_bit(x: float, n: int, inte: int, frac: int):
    x_mp = mpfr(abs(x)) * pow(2, frac)
    return int(x_mp) >> (inte - n)

def extract_int(x: float):
    x_mp = mpfr(abs(x))
    i = trunc(x_mp)
    return 0 if i == 0 else int(log2(i)) + 1

def comp_first_n_bit(path: Path, n: int):
    bold(f"\nComparing first {n} bits for: {path.name}\n")
    files = retrieveFiles(path)
    fv = re.search(r"Values Begin\n([\s\S]*?)\nValues End", files[0]).group(1)
    tv = re.search(r"Values Begin\n([\s\S]*?)\nValues End", files[1]).group(1)
    max_int = max(extract_int(float(v)) for v in fv.splitlines())
    print(f"max int bits: {max_int}", flush=True)
    for fv_line, tv_line in zip(fv.splitlines(), tv.splitlines()):
        # skip empty lines
        if not fv_line.strip() and not tv_line.strip():
            continue
        fv_f, tv_f = float(fv_line), float(tv_line)
        fn = extract_first_n_bit(fv_f, n, 128, 128 - max_int)
        tn = extract_first_n_bit(tv_f, n, 128, 128 - max_int)
        if abs(fn - tn) > 1:
            print(f"{fv_f} != {tv_f} -> {fn:0{n}b} != {tn:0{n}b}", flush=True)

if __name__ == '__main__':
    debug = False
    parser = argparse.ArgumentParser(description='Taffo test runner')
    parser.add_argument('-common-args', type=str, default='', help='Extra flags to pass to both float and taffo compilations')
    parser.add_argument('-tests-dir', type=str, default='.', help='Root dir containing test folders')
    parser.add_argument('-only', type=str, help='Comma-separated list of benchmark names')
    parser.add_argument('-clean', action='store_true', help='Clean benchmarks')
    parser.add_argument('-init', action='store_true', help='Init benchmarks')
    parser.add_argument('-fullreset', action='store_true', help='Reset benchmarks')
    parser.add_argument('-compile', action='store_true', help='Compile benchmarks')
    parser.add_argument('-run', action='store_true', help='Run benchmarks')
    parser.add_argument('-validate', action='store_true', help='Validate benchmarks')
    parser.add_argument('-comp_int', type=int, default=0, help='Compare first n bits')
    parser.add_argument('-ordereddiff', action='store_true', help='Show ordered diffs by error')
    parser.add_argument('-debug', action='store_true', help='Enable debug build')
    parser.add_argument('-plot_compile_time', action='store_true', help='Plot compile times')
    parser.add_argument('-diff_only', action='store_true', help='Only diff outputs instead of complete validation')
    args = parser.parse_args()

    common_args = ""
    if args.common_args:
        common_args = args.common_args

    tests_dir = Path(args.tests_dir)

    if args.only:
        names = set(args.only.split(','))
        only  = []
        # for each name the user asked, find its directory anywhere under tests_dir
        for name in names:
            matches = [
                p for p in tests_dir.rglob(name)
                if p.is_dir() and ((p/f"{name}.c").exists() or (p/f"{name}.cpp").exists())
            ]
            if not matches:
                print(f"Warning: no benchmark directory found for '{name}'", file=sys.stderr)
            else:
                only.extend(matches)
    else:
        only = [
            p for p in tests_dir.rglob('*')
            if p.is_dir() and ((p/f"{p.name}.c").exists() or (p/f"{p.name}.cpp").exists())
        ]

    # compute padding for aligned status output
    labels = []
    for p in only:
        nm = p.name
        labels += [
            f"Compiling: {nm}-float",
            f"Compiling: {nm}-taffo",
            f"Running: {nm}"
        ]
    ACTION_PAD = max(len(lbl) for lbl in labels) + 1

    if args.fullreset:
        for p in only:
            clean(p, ('.c', '.cpp', '.ll', '.py', '.txt'))
        sys.exit(0)

    if args.clean:
        for p in only:
            clean(p, ('.c', '.cpp', '.ll', '.py', '.txt', '.h', '.hpp'))
        sys.exit(0)

    if args.plot_compile_time:
        bold("PLOT COMPILE TIME")
        datas = [extract_compile_time(p) for p in only]
        plot_compile_times(tests_dir, datas)
        sys.exit(0)

    if args.ordereddiff:
        for p in only:
            ordereddiff(p)
        sys.exit(0)

    if args.init:
        for p in only:
            generatedata(p)

    do_compile  = args.compile  or not (args.run     or args.validate)
    do_run      = args.run      or not (args.compile or args.validate)
    do_validate = args.validate or not (args.compile or args.run)
    debug = args.debug

    # 1) COMPILE
    compile_ok = set()
    if do_compile:
        bold("COMPILE")
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
            results = list(ex.map(compile, only))
        for path, ok, log in results:
            print(log, end="", flush=True)
            if ok:
                compile_ok.add(path)
    else:
        compile_ok = set(only)

    # 2) RUN (only those that compiled)
    run_ok = set()
    if do_run:
        bold("RUN")
        for p in only:
            if p not in compile_ok:
                lbl = f"Running: {p.name}".ljust(ACTION_PAD)
                print(f"{lbl}{Fore.YELLOW}{Style.BRIGHT}SKIP (compile failed){Style.RESET_ALL}", flush=True)
                continue
            ok = run(p)
            if ok:
                run_ok.add(p)
    else:
        run_ok = set(only)

    # 3) VALIDATE (only those that ran)
    if do_validate:
        bold("VALIDATE")
        if args.diff_only:
            rows = []
            for p in only:
                if p not in run_ok:
                    rows.append({"name":p.name, "correct": False})
                else:
                    for suf, (f_txt,t_txt) in retrieveFiles(p).items():
                        rows.append({"name":p.name+suf, "correct":(f_txt==t_txt)})
            df = pd.DataFrame(rows)
            print(df.to_string(index=False, columns=["name","correct"]), flush=True)
        else:
            allres = []
            for p in only:
                if p not in run_ok:
                    lbl = f"{p.name}".ljust(ACTION_PAD)
                    print(f"{lbl}{Fore.YELLOW}{Style.BRIGHT}SKIP (run failed){Style.RESET_ALL}", flush=True)
                else:
                    allres.extend(validate(p, compute_speedup=True))
            if allres:
                print_tables(allres)
            if args.comp_int > 0:
                for b in (4,8,16):
                    for p in only:
                        if p in run_ok:
                            comp_first_n_bit(p, b)
