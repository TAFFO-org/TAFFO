#!/usr/bin/env python3

from glob import glob
import sys
import os

def rename():
    full_path = os.path.dirname(os.path.realpath(__file__))
    for bench in glob(full_path + "/output/*"):
        for file_path in [ x for x in glob( bench+"/taffo/*.log") if "stm32" not in x ]:
            name = file_path.split("/")[-1]
            splitted = name.split("_")
            if len(splitted)==4:
                name = f"{splitted[0]}_{int(splitted[1]):04}_{int(splitted[2]):04}_{int(splitted[3][:-4]):04}.log"
                os.rename(file_path,bench+"/taffo/"+name)



if __name__ == "__main__":
    rename()