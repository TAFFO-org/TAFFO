#!/usr/bin/env python3

from glob import glob
import rename
import os
import sys

import subprocess
def clean(path):
    subprocess.run(f"{path}/build.sh clean", shell=True)
    subprocess.run(f"cd {path}/embedded_src/; make clean", shell=True)
    subprocess.run(f"mkdir {path}/embedded_src/bench_obj", shell=True)


def write_c_file(path, bench):
    if "orig" in bench:
        name  = "bench_" + bench.split("/")[-1].split(".")[0] + "_orig"
    else:
        name = bench.split("/")[-1].split(".")[0]
    with open(f"{path}/embedded_src/bench_main.h", "w") as file:
        file.write(f"void {name}();\n")
    with open(f"{path}/embedded_src/bench_main.c.in", "w") as file:
        file.write(f"printf(\"%s\\n\",\"{name}\");\n")
        file.write(f"{name}();\n")

def copy_file(path, bench):
    if "orig" in bench:
        start = "/".join(bench.split("/")[:-1])
        files = glob(f"{start}/*.o")
        for file in files:
            subprocess.run(f"cp {file} {path}/embedded_src/bench_obj", shell=True)        
    else:
        subprocess.run(f"cp {bench} {path}/embedded_src/bench_obj", shell=True)

def make_file(path, bench):
    out = subprocess.run(f"cd {path}/embedded_src; make", shell=True)
    if out.returncode != 0:
        print(f"{out.returncode} ERROR!!")
        exit()
    out = subprocess.run(f"cd {path}/embedded_src; make flash", shell=True)
    if out.returncode != 0:
        print(f"{out.returncode} ERROR!!")
        exit()
    out = subprocess.run(f"sleep 1; cd {path}/embedded_src; make monitor", shell=True)
    if out.returncode != 0:
        print(f"{out.returncode} ERROR!!")
        exit()


def  copy_result(path, output_root, bench):
    final_destination = output_root
    name_bench = bench.split("/")[-3]
    if "orig" in bench:
        name_log = bench.split("/")[-1].split(".")[0] + "_float.log"
        
        final_destination = final_destination + f"/{name_bench}/float/{name_log}" 
    else:
        name_log = bench.split("/")[-1].split(".")[0][6:] + ".log"
        name_log = name_log.replace("_stm32f4_float", "")
        name_log = name_log.replace("_stm32_float", "")
        final_destination = final_destination + f"/{name_bench}/taffo/{name_log}"        

    out = subprocess.run(f"mv {path}/embedded_src/monitor.log {final_destination}", shell=True)



if __name__ == "__main__":
  rename.rename()
  full_path = os.path.dirname(os.path.realpath(__file__))
  out_folder_root = full_path + "/output"
  if len(sys.argv) > 1:
    out_folder_root = sys.argv[1]

  for bench in glob(full_path + "/output/*"):   
    """"Load Taffo Bench"""
    for file_path in [ x for x in glob( bench+"/**/*.o")]:
        last = file_path.split("/")[-1]
        if "data" in last:
            continue
        if "01" in last:
            continue

        clean(full_path)
        write_c_file(full_path,file_path)
        copy_file(full_path, file_path)
        make_file(full_path, file_path)
        copy_result(full_path, out_folder_root, file_path)
        