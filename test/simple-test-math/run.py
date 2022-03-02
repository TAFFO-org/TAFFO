#!/usr/bin/python3
import argparse
import glob
import subprocess
import sys
import os



def remove_files(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def up_one():
    sys.stdout.buffer.write(b"\x1B[1F")

def print_byte(text):
    if isinstance(text,str) :
        sys.stdout.buffer.write(text.encode("ASCII"))
    else:
        sys.stdout.buffer.write(text)
    sys.stdout.flush()


def compile(names):
    command = "taffo {0}.{1} -temp-dir ./temp -o ./build/{0}_taffo -float-output ./build/{0} -debug -lm"
    for name, ext in names:
        print_byte("[...] Compiling {0}\n".format(name))
        up_one()
        with open("./log/{0}".format(name), "w") as f:
            res = subprocess.run(command.format(name, ext), stdout=f, stderr=subprocess.STDOUT , shell=True)
        if res.returncode == 0:
            print_byte("[OKK] Compiling {0}\n".format(name))
        else:
            print_byte("[ERR] Compiling {0}\n".format(name))

       
def run(names):
    command = "./build/{}"
    for name in names:
        print_byte("[...] Running {0}\n".format(name))
        up_one()
        with open("./output/{0}".format(name), "w") as f:
            res = subprocess.run(command.format(name), stdout=f, stderr=subprocess.STDOUT , shell=True)
        if res.returncode == 0:
            print_byte("[OKK] Running {0}\n".format(name))
        else:
            print_byte("[ERR] Running {0}\n".format(name))


def validate(names):

    for name in names:
        print(name)
        try:
            with open("./output/"+name , "r") as orig_file, open("./output/"+name+"_taffo" , "r") as taffo_file:
                tot_error=0.0
                count = 0
                
                for orig_line, taffo_line in zip(orig_file.readlines(), taffo_file.readlines()):
                    orig_float = float(orig_line)
                    taffo_float = float(taffo_line)
                    tot_error = tot_error + abs(orig_float-taffo_float)
                    count = count + 1

            print("tot_error, lines, avg_error")
            print("{}, {}, {}".format(tot_error, count, tot_error/count ))
        except:
            print("Error")

        print("\n\n\n\n")






    



def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("-only", help="test only this file")
    parser.add_argument("-compile" , action='store_true' , help="only compile")
    parser.add_argument("-run" , action='store_true' ,help="runs only")
    parser.add_argument("-validate" , action='store_true' , help="validate only")
    parser.add_argument("-clean" , action='store_true' , help="clean all")
    args = parser.parse_args()




    bcompile = False
    brun = False
    bvalidate = False

    if args.clean:
        remove_files("./output")
        remove_files("./build")
        remove_files("./log")
        remove_files("./temp")
        return 0


    if args.compile:
        bcompile = True
    if args.run:
        brun = True
    if args.validate:
        bvalidate = True


    if (not args.compile and not args.run and not args.validate):
        bcompile = True
        brun = True
        bvalidate = True


    if bcompile:
        if args.only:
            names = [args.only]
        else:
            names = [x for x in glob.glob("*.c")] + [x for x in glob.glob("*.cpp")]
        names = [(".".join(x.split('.')[:-1]), x.split('.')[-1:]) for x in names]
        compile(names)

    if brun:
        if args.only:
            names = [".".join(args.only.split(".")[:-1])] + [".".join(args.only.split(".")[:-1]) + "_taffo"]
        else:
            names = [x for x in glob.glob("./build/*")]
        names = ["".join(x.split("/")[-1:]) for x in names]
        run(names)

    if bvalidate:
        if args.only:
            names = [".".join(args.only.split(".")[:-1])]
        else:
            names = [x for x in glob.glob("./output/*") if "taffo" not in x]
        names = ["".join(x.split("/")[-1:]) for x in names]
        validate(names)





if __name__ == "__main__":
    main()
 













