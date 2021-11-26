# AxBench TAFFO Benchmarks

http://axbench.org/

TAFFO must have been compiled and installed, and its binaries must be in the current `$PATH`.

First, do
```bash
$ export LLVM_DIR=/path/to/LLVM/install
$ source ./setenv.sh /path/to/TAFFO/install
```

Then, you may enter each one of the app directories, and compile it with
```bash
$ ./compile+collect.sh
```
Export `ENABLE_ERROR=1` to enable error propagation.

Run the benchmarks with
```bash
$ ./run.sh
```
or
```bash
$ ./run2.sh
```
for a more machine-readable output.
