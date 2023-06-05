# Polybench OpenMP TAFFO Benchmarks
Version 3.2 adapted for OpenMP computation

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

## Modifications
Note that most benchmarks in [the original repository](https://github.com/cavazos-lab/PolyBench-ACC) have been modified, not only by adding annotations, but by fixing syntactical errors in the usage of OpenMP.

A not-negligible part of the benchmarks has not only syntactical, but also logical errors, when compiling with plain clang. Therefore, they have been excluded from compilation and testing.
The disabled benchmarks, with the related motivation, are the following:
- datamining/correlation - Segmentation Fault
- linear-algebra/kernels/atax - Segmentation Fault
- linear-algebra/kernels/bicg - Segmentation Fault
- linear-algebra/kernels/cholesky - Segmentation Fault
- linear-algebra/kernels/gemver - Segmentation Fault
- linear-algebra/kernels/gesummv - Segmentation Fault
- linear-algebra/kernels/mvt - Segmentation Fault
- linear-algebra/kernels/trisolv - Segmentation Fault
- linear-algebra/kernels/trmm - Segmentation Fault
- linear-algebra/solvers/durbin - Segmentation Fault
- linear-algebra/solvers/dynprog - Segmentation Fault
- linear-algebra/solvers/gramschmidt - Segmentation Fault
- linear-algebra/solvers/lu - The results of the benchmark are mostly NaN
- linear-algebra/solvers/ludcmp - The results of the benchmark are mostly NaN
- stencils/adi - Although the benchmark should output numbers between 0 and 1, the resulting numbers are extremely high
- stencils/convolution-2d - Segmentation Fault
- stencils/convolution-3d - Segmentation Fault
- stencils/fdtd-apml - Segmentation Fault

You can find a list of the working ones in the [benchmark_list file](./utilities/benchmark_list).

A complete diff between the original OpenMP PolyBench and the current TAFFO annotated version can be found in [MODIFICATIONS.patch](./MODIFICATIONS.patch).
