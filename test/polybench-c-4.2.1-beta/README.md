# Polybench TAFFO Benchmarks
Version 4.2.1

http://web.cse.ohio-state.edu/~pouchet.2/software/polybench/

## Usage

TAFFO must have been compiled and installed, and its binaries must be in the current `$PATH`.

First, do
```bash
$ export LLVM_DIR=/path/to/LLVM/install
```

To compile, run, and validate (i.e., obtain execution times and errors) all benchmarks run
```bash
$ ./collect-fe-stats.sh <OUT-DIR> <REPS>
```
`<OUT-DIR>` is the output directory for the results of validation,
and `<REPS>` is the number of times each benchmark should be executed
(the average execution time is then computed among them).


## Alternative usage

Otherwise, you may execute such steps separately.
Compile all benchmarks with
```bash
$ ./compile.sh
```
Run the benchmarks with
```bash
$ ./run.sh
```

Then, the time and error results can be obtained with
```bash
$ ./validate.py
```

All such scripts can be run with the option `--only=path/to/benchmark.c`
to process one single benchmark instead of all of them.
