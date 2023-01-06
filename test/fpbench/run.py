#!/usr/bin/python3
import argparse
from pathlib import *
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import sys
import gmpy2
from gmpy2 import mpfr, trunc, log2
gmpy2.get_context().precision=100

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

def bold( s : str):
    sys.stdout.buffer.write(b"\x1B\x5B1m")
    print(s, end="")
    flush()
    sys.stdout.buffer.write(b"\x1B\x5Bm\n")

def print_okk():
    sys.stdout.buffer.write(b"\x1B\x5B32mOKK!\x1B\x5Bm\n")
    flush()

def print_err():
    sys.stdout.buffer.write(b"\x1B\x5B31mERR!\x1B\x5Bm\n")
    flush()

def flush():
    sys.stdout.flush()

def generatedata(path : Path):
    subprocess.run("cd {}; ./datagenerator.py > data.h".format(path.as_posix()), shell=True)

def compiletaffo(path: Path):
    global debug, common_flags
    bench_name = path.name + ".c"
    bench_exec = path.name + "-taffo"
    pipe_out = subprocess.DEVNULL
    compile_flag = f"{common_flags} -time-profile-file {path.absolute().as_posix()}/{path.name}_taffo_time.csv"
    if debug:
        (path / "./llvm-file").mkdir(parents=True, exist_ok=True)
        compile_flag = f"{compile_flag} -debug -temp-dir ./llvm-file"
        pipe_out = open(f"{path.as_posix()}/{path.name}_taffo.log", "w")


    print("Compiling: {}\t".format(bench_exec), end="")
    flush()
    s = subprocess.run("cd {}; taffo {} {} -o {} -lm".format(path.as_posix(), compile_flag, bench_name, bench_exec), shell=True , stdout=pipe_out, stderr=pipe_out)
    if s.returncode == 0:
        print_okk()
    else:
        with open(f"./{path.as_posix()}/{path.name}_taffo.log") as file:
            print(file.read())
        print_err()
    if pipe_out == subprocess.DEVNULL and debug:
        pipe_out.close()


def compilefloat(path: Path):
    global debug, common_flags
    compile_flag = f"{common_flags}"
    pipe_out = subprocess.DEVNULL
    if debug:
        compile_flag = f"{common_flags}"
        pipe_out = open(f"{path.as_posix()}/{path.name}_float.log", "w")
    bench_name = path.name + ".c"
    bench_exec = path.name + "-float"
    print("Compiling: {}\t".format(bench_exec), end="")
    flush()
    s = subprocess.run("cd {}; clang-12 {}   {} -o {}  -lm &> /dev/null".format(path.as_posix(), compile_flag ,bench_name, bench_exec), shell=True,  stdout=pipe_out, stderr=pipe_out)

    if s.returncode == 0:
        print_okk()
    else:
        print_err()


def compile(path :Path):  
    if not (path / "data.h").exists():
        generatedata(path)
    compilefloat(path)
    compiletaffo(path)


def init(path :Path):
    generatedata(path)


def run(path :Path):
    bench_name = path.name
    print("Running: {}\t".format(bench_name), end="")
    flush()
    if not ((path / "{}-taffo".format(bench_name)).exists() and (path / "{}-float".format(bench_name)).exists()):
        print_err()
    else:
        
        s = subprocess.run("cd {};  ./{}-taffo > taffo-res".format(path.as_posix(), bench_name,), shell=True)
        s1 = subprocess.run("cd {};  ./{}-float > float-res".format(path.as_posix(), bench_name,), shell=True)
        if s.returncode != 0:
            print_err()
        elif s1.returncode != 0:
            print_err()
        else:
            print_okk()

    

def retriveFiles(path: Path):
    bench_name = path.name
    if not ((path / "taffo-res").exists() and (path / "float-res").exists()):
        print_err()
    content = []
    with open(path / "float-res", "r") as file:
        content.append(file.read())
    with open(path / "taffo-res", "r") as file:
        content.append(file.read())
    return content

def retriveCompilationTime(path: Path):
    bench_name = path.name
    time_file = (path / f"{bench_name}_taffo_time.csv")
    if not (time_file.exists()):
        print_err()
    content = pd.read_csv(time_file, sep=',')
    return content
    
def reject_outliers(data):
    data = np.array(data)
    mean = np.mean(data)
    standard_deviation = np.std(data)
    distance_from_mean = abs(data - mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    return data[not_outlier]

def getTime(file: str):
    times_float = [float(x) for x in re.findall(r"Cycles: (\d+)",file[0])]
    avg_float_time = np.mean(reject_outliers(times_float))
    times_taffo = [float(x) for x in re.findall(r"Cycles: (\d+)",file[1])]
    avg_taffo_time = np.mean(reject_outliers(times_taffo))
    return (avg_float_time, avg_taffo_time)

def getData(file: str):
    float_values = re.findall(r"Values Begin\n([\s\S]*)\nValues End",file[0])[0]
    taffo_values = re.findall(r"Values Begin\n([\s\S]*)\nValues End",file[1])[0]
    abs_err = []
    rel_err = []
    max_abs = 0
    max_rel = 0
    for float_value,taffo_value in zip(float_values.split("\n"),taffo_values.split("\n")):
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
    return {"rel_err" : rel_err,  "max_rel" : max_rel, "abs_err": abs_err, "max_abs" : max_abs }


def ordereddiff(path :Path):
    files= retriveFiles(path)
    float_values = re.search(r"Values Begin\n([\s\S]*)\nValues End",files[0])
    taffo_values = re.search(r"Values Begin\n([\s\S]*)\nValues End",files[1])
    print(path)
    f = files[0][0 : float_values.start()].count("\n") +2
    s = files[1][0 : taffo_values.start()].count("\n") +2
    if f != s:
        print_err()
        return
    rel_err = []
    i = 0
    for float_value,taffo_value in zip(float_values.group(1).split("\n"),taffo_values.group(1).split("\n")):
        float_value = float(float_value)
        taffo_value = float(taffo_value)
        if (float_value == 0 or taffo_value == 0):
            i+=1
            continue

        tmp = abs(taffo_value-float_value)
        rel_err.append((f+i, tmp/abs(float_value)))
        i = i +1

    rel_err.sort(reverse=True, key= lambda x : x[1])
    
    
    for x,y in rel_err:
        print("{}, {}".format(x,y))


    

    
    

def validate(path :Path):
    bench_name = path.name
    files= retriveFiles(path)
    compile_breakdown = retriveCompilationTime(path)
    times = getTime(files)
    datas = getData(files)
    speedup = (times[0] /times[1])
    compile_time = compile_breakdown.iloc[0]['taffo_end'] - compile_breakdown.iloc[0]['taffo_start']
    ret = {"name" : bench_name, "speedup" : speedup, "compile_time": compile_time}
    ret.update(datas)
    return ret

def extract_compile_time(path :Path):
    bench_name = path.name
    compile_breakdown = retriveCompilationTime(path)
    ret = {
        "name" : bench_name,
        "total_time": compile_breakdown.iloc[0]['taffo_end'] - compile_breakdown.iloc[0]['taffo_start'],
        "base_ll_time": compile_breakdown.iloc[0]['init_start'] - compile_breakdown.iloc[0]['taffo_start'],
        "backend_time": compile_breakdown.iloc[0]['taffo_end'] - compile_breakdown.iloc[0]['backend_start'],
        "taffo_init_time": compile_breakdown.iloc[0]['vra_start'] - compile_breakdown.iloc[0]['init_start'],
        "vra_time": compile_breakdown.iloc[0]['dta_start'] - compile_breakdown.iloc[0]['vra_start'],
        "dta_time": compile_breakdown.iloc[0]['conversion_start'] - compile_breakdown.iloc[0]['dta_start'],
        "conversion_time": compile_breakdown.iloc[0]['backend_start'] - compile_breakdown.iloc[0]['conversion_start'],
    }
    return ret

def plot_compile_times(table):
    table = pd.DataFrame(table)
    abs_plot_table = table.drop('total_time', axis=1)
    ax = abs_plot_table.plot.barh(x='name', stacked=True, title='TAFFO compilation stages time', figsize=(10, 8))
    fig = ax.get_figure()
    plt.xlabel("Seconds")
    plt.ylabel("Benchmark")
    fig.savefig(f'compile_time.png', dpi=fig.dpi, bbox_inches = 'tight')

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
    fig2.savefig(f'compile_time_rel.png', dpi=fig2.dpi, bbox_inches = 'tight')

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
    fig3.savefig(f'compile_time_taffo_rel.png', dpi=fig3.dpi, bbox_inches = 'tight')
    print(rel_taffo_plot_table)


def clean(path :Path, suffix : tuple):
    bench_name = path.name
    files= [x for x in path.glob("*") if not x.name.endswith(suffix) ]
    for f in files:
        if f.is_file():
            f.unlink()
        if f.is_dir():
            clean(f, (".cc"))
            f.rmdir()

    
def print_tables(table):
    table = pd.DataFrame(table)
    print(table.to_string(columns=["name", "speedup", "compile_time", "rel_err", "abs_err"], formatters={"rel_err": '{:,.8%}'.format, "max_rel": '{:,.2f}'.format }))


def extract_first_n_bit( x : float, n : int, inte : int, frac : int):

    x = mpfr(x)
    if x < 0:
        x = x * -1        
    x = x * pow(2, frac)
    x = int(x)
    x=x >> (inte-n) 
    return(x)

def extract_int(x: float):
    if x < 0 :
        x = x * -1
    x = mpfr(x)
    x = trunc(x)
    if x == 0:
        return 0
    return int(log2(x)) + 1

def comp_first_n_bit(path :Path, n : int):
    bench_name = path.name
    print("\nComparing : {}\n".format(bench_name), end="")
    files= retriveFiles(path)
    float_values = re.findall(r"Values Begin\n([\s\S]*)\nValues End",files[0])[0]
    taffo_values = re.findall(r"Values Begin\n([\s\S]*)\nValues End",files[1])[0]
    max_int_size = 0
    for float_value in float_values.split("\n"):
        max_int_size = max(max_int_size, extract_int(float(float_value)))
    print(f"max int: {max_int_size}")
    for float_value,taffo_value in zip(float_values.split("\n"),taffo_values.split("\n")):
        float_value = float(float_value)
        taffo_value = float(taffo_value)
        float_n_bit = extract_first_n_bit(float_value,n,128, 128-max_int_size)
        taffo_n_bit =  extract_first_n_bit(taffo_value,n,128, 128-max_int_size)

        if  float_n_bit + 1 < taffo_n_bit or float_n_bit - 1 > taffo_n_bit:
            print(("{} != {} -> {:0"+str(n)+"b} != {:0"+str(n)+"b}").format(float_value,taffo_value, float_n_bit, taffo_n_bit))




if __name__ == '__main__':
    debug = False
    parser = argparse.ArgumentParser(description='FPbench runner')
    parser.add_argument('-M', metavar='integer', type=int, nargs='?',
                        help='Number of Iteration', default=10000)
    parser.add_argument('-only', metavar='name[,name]*', type=str,
                        help='List of comma separated benchmark\'s names')
    parser.add_argument('-clean', metavar='bool', type=bool, default=False, nargs='?', help='Clean Benchmarks' , const=True)
    parser.add_argument('-init', metavar='bool', type=bool, default=False, nargs='?', help='Init Benchmarks' , const=True)
    parser.add_argument('--fullreset', metavar='bool', type=bool, default=False, nargs='?', help='Reset Benchmarks' , const=True)
    parser.add_argument('-compile', metavar='bool', type=bool, default=False, nargs='?', help='Compile Benchmarks' , const=True)
    parser.add_argument('-run', metavar='bool', type=bool, default=False, nargs='?', help='Run Benchmarks',   const=True)
    parser.add_argument('-validate', metavar='bool', type=bool, default=False, nargs='?', help='Validate Benchmarks', const=True)
    parser.add_argument('-comp_int', metavar='int', type=int, default=0, nargs='?', help='Compare first n bit', const=True)
    parser.add_argument('-ordereddiff', metavar='bool', type=bool, default=False, nargs='?', help='Print out an ordered list of line, error sorted by max error', const=True)
    parser.add_argument('-debug', metavar='bool', type=bool, default=False, nargs='?', help='debug build', const=True)
    parser.add_argument('-plot_compile_time', metavar='bool', type=bool, default=False, nargs='?', help='plot compilation time of benchmarks', const=True)


    args = parser.parse_args()
    M = args.M
    common_flags = f"-O3 -DAPP_MFUNC -DM={M} -fno-vectorize -fno-slp-vectorize"
    if args.only is None:
        only = [x for x in Path(".").glob("*/") if x.is_dir() and "data" not in x.name]
    else:
        only = [Path("./" + x) for x in args.only.split(",")]
    
    bcompile = args.compile
    bvalidate = args.validate
    brun = args.run
    debug = args.debug
    if (bcompile or bvalidate or brun) == False :
        bcompile=True
        brun=True
        bvalidate=True

    if args.fullreset:
        for path in only:
            clean(path, (".c", ".py"))
        exit(0)

    if args.clean:
        for path in only:
            clean(path, (".c", ".py", ".h"))
        exit(0)

    if args.plot_compile_time:
        bold("PLOT COMPILE TIME")
        datas = []
        for path in only:
            datas.append(extract_compile_time(path))
        plot_compile_times(datas)
        exit(0)

    if args.ordereddiff:        
        for path in only:
            ordereddiff(path)
        exit(0)

    if args.init:
        for path in only:
            init(path)
    

    if bcompile:
        bold("COMPILATION")  
        for path in only:
            compile(path)

    if brun:
        bold("RUN")  
        for path in only:
            run(path)

    if bvalidate:
        bold("VALIDATE") 
        datas = []
        for path in only:
            datas.append(validate(path))
        print_tables(datas)
        if args.comp_int > 0:
            for path in only:
                print("4 bits")
                comp_first_n_bit(path, 4)
            for path in only:
                print("8 bits")
                comp_first_n_bit(path, 8)
            for path in only:
                print("16 bits")
                comp_first_n_bit(path, 16)


    


    


    