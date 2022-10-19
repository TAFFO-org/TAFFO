#!/usr/bin/python3
import argparse
from pathlib import *
import subprocess
import re
import numpy as np
import pandas as pd
import sys
import gmpy2
from gmpy2 import mpfr, trunc, log2
gmpy2.get_context().precision=100


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
    compile_flag = f"{common_flags}"
    if debug:
        (path / "./llvm-file").mkdir(parents=True, exist_ok=True)
        compile_flag = f"{common_flags} -debug -temp-dir ./llvm-file"
        pipe_out = open(f"{path.as_posix()}/{path.name}_taffo.log", "w")


    print("Compiling: {}\t".format(bench_exec), end="")
    flush()
    s = subprocess.run("cd {}; taffo {} {} -o {} -lm".format(path.as_posix(), compile_flag, bench_name, bench_exec), shell=True , stdout=pipe_out, stderr=pipe_out)
    if s.returncode == 0:
        print_okk()
    else:
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
        if f.is_file():
            f.unlink()
        if f.is_dir():
            clean(f, (".cc"))
            f.rmdir()

    
def print_tables(table):
    table = pd.DataFrame(table)
    print(table.to_string(columns=["name", "speedup", "rel_err", "abs_err"], formatters={"rel_err": '{:,.8%}'.format, "max_rel": '{:,.2f}'.format }))


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
    common_flags = "-O3 -DAPP_MFUNC -DM=100000 -fno-vectorize -fno-slp-vectorize"
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


    args = parser.parse_args()
    M = args.M
    if args.only is None:
        only = [x for x in Path(".").glob("*/") if x.is_dir()]
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


    


    


    