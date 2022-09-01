#!/usr/bin/env python
import argparse
from pathlib import *
import subprocess
import re
import numpy as np
import pandas as pd
import sys

def bold( s : str):
    sys.stdout.buffer.write(b"\x1B\x5B1m")
    print(s, end="")
    flush()
    sys.stdout.buffer.write(b"\x1B\x5Bm\n")

def print_okk():
    sys.stdout.buffer.write(b"\x1B\x5B32mOKK!\x1B\x5Bm\n")

def print_err():
    sys.stdout.buffer.write(b"\x1B\x5B31mERR!\x1B\x5Bm\n")

def flush():
    sys.stdout.flush()

def generatedata(path : Path):
    subprocess.run("cd {}; ./datagenerator.py > data.h".format(path.as_posix()), shell=True)

def compiletaffo(path: Path):
    bench_name = path.name + ".c"
    bench_exec = path.name + "-taffo"
    print("Compiling: {}\t".format(bench_exec), end="")
    flush()
    s = subprocess.run("cd {}; taffo -O3 {} -o {} -lm".format(path.as_posix(), bench_name, bench_exec), shell=True , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if s.returncode == 0:
        print_okk()
    else:
        print_err()

def compilefloat(path: Path):
    bench_name = path.name + ".c"
    bench_exec = path.name + "-float"
    print("Compiling: {}\t".format(bench_exec), end="")
    flush()
    s = subprocess.run("cd {}; gcc -O3 {} -o {} -lm &> /dev/null".format(path.as_posix(), bench_name, bench_exec), shell=True,  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        tmp = abs(taffo_value-float_value)
        abs_err.append(tmp)
        rel_err.append(tmp/abs(float_value))
        max_abs = max(max_abs, tmp)
        max_rel = max(max_rel, tmp/abs(float_value))
    abs_err = np.mean(abs_err)
    rel_err = np.mean(rel_err)
    return {"rel_err" : rel_err,  "max_rel" : max_rel, "abs_err": abs_err, "max_abs" : max_abs }

    

def validate(path :Path):
    bench_name = path.name
    files= retriveFiles(path)
    times = getTime(files)
    datas = getData(files)
    speedup = (times[0] /times[1])
    ret = {"name" : bench_name, "speedup" : speedup}
    ret.update(datas)
    return ret


def clean(path :Path, suffix : tuple):
    bench_name = path.name
    files= [x for x in path.glob("*") if not x.name.endswith(suffix) ]
    for f in files:
        f.unlink()

    
def print_tables(table):
    table = pd.DataFrame(table)
    print(table)



if __name__ == '__main__':
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
    args = parser.parse_args()
    M = args.M
    if args.only is None:
        only = [x for x in Path(".").glob("*/") if x.is_dir()]
    else:
        only = [Path("./" + x) for x in args.only.split(",")]
    
    bcompile = args.compile
    bvalidate = args.validate
    brun = args.run
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
    exit(0)
    


    


    