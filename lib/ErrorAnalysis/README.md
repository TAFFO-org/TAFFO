# TAFFO Error Propagator for Fixed Point Computations

An LLVM opt pass that computes errors propagated in programs containing fixed point computations.

## Build and run

This pass requires LLVM 6.0.1 to be built and installed.
Build with:
```
$ cd taffo-error
$ mkdir build
$ cd build
$ LLVM_DIR="/path/to/llvm/install" cmake ..
$ make
```

Then, the pass can be run with
```
$ opt -load /path/to/taffo-error/build/ErrorPropagator/libLLVMErrorPropagator.so -errorprop source.ll
```

You may run the lit regression tests with
```
$ llvm-lit /path/to/taffo-error/build/test/ErrorPropagator
```

## Usage

TAFFO Error Propagator (TAFFO-EP) will propagate initial errors and rounding errors for all instructions whose operands meet the following requirements:
- There must be a range attached to them by means of input info metadata (cf. `Metadata.md`);
  this requirement is relaxed for instructions that do not (usually) change the range of their single operand,
  such as extend and certain conv instructions.
- An absolute error for them must be available, either because it has been propagated by this pass, or because it is attached to them as initial error in input info metadata.
  If both error sources are available, TAFFO gives precedence to the computed error.
  Whenever the initial error is used for an instruction, TAFFO-EP emits a warning as a debug message.

Moreover, TAFFO-EP needs type info, i.e. bit width and number of fractional bits of instructions and global variables in order to compute truncation errors for fixed point operations.
Type info, range and initial error may be attached to instructions and global variables as input info metadata.
Information for formal function parameters must be attached to their function.
Further documentation about the format of metadata and the APIs for setting and accessing them may be found in `Metadata.md`.
Note that when a range or an initial error are attached to arrays or pointers to arrays, they are considered valid and equal for each element of the array
(one may still inspect the absolute errors attached to intermediate instructions in case elements of an array are not used homogeneously).

The relative error computed for each instruction is attached to it as metadata.
Moreover, it is possible to mark some instructions or global variables as targets: TAFFO-EP will keep track of their relative errors, and display it at the end of the pass (see `Metadata.md`).

An important caveat: TAFFO-EP uses Alias Analysis to retrieve errors associated to the values loaded by `load` instructions.
For it to function properly, the input LLVM IR file must be in proper SSA form.
Therefore, the `-mem2reg` pass should be scheduled before this pass.

### Command line argumments

- `-startonly`: only propagate errors for functions marked as starting points with the appropriate metadata (cf. `Metadata.md`).
  This option is useful when dealing with large programs, in which the errors of only a few functions must be evaluated.
  Output error metadata for instructions in functions not marked as starting points are not produced.
  A function that is not a starting point may still be evaluated if it is called (possibly indirectly) by a starting-point function.
- `-recur <num>`: default number of recursive calls allowed for each function.
  For mutually recursive functions, each one of them is executed at most `<num>` times.
  The default value is 1.
  The maximum recursion threshold may be set for a specific function with metadata attached to it (cf. `Metadata.md`), overriding the default value set by this option.
- `-cmpthresh <perc>`: if this option is set, metadata will be emitted for `cmp` instructions whose operand has an absolute error greater than the tolerance for this comparison by `<perc>`%.
  The tolerance for a comparison is the minimum error on the operand necessary for making the comparison wrong.
  The default value of `<perc>` is 0 (i.e. a comparison error is signaled every time it is deemed possible).
- `-dunroll <trip>`: default loop unroll count.
- `-nounroll`: never unroll loops.
- `-relerror`: output relative errors instead of absolute errors (experimental).
- `-exactconst`: treat all constants as exact (do not add rounding error).

### Loop Unrolling

In order to correctly bound errors in iterative computations, TAFFO-EP can unroll loops by means of the LLVM loop unrolling facilities.
The number of times a loop is unrolled (the trip count) is determined from the following sources, in decreasing order of priority:
1. the trip count detected by LLVM function `unsigned ScalarEvolution::getSmallConstantTripCount(const Loop *L)`;
2. the unroll count specified by metadata attched to the terminator instruction of the loop header (cf. `Metadata.md`);
3. the default unroll count specified with command line option `-dunroll`.

The LLVM loop unrolling facilities need loops to be normalized.
Therefore, before running TAFFO-EP the following optimization passes should be scheduled:
`-mem2reg -simplifycfg -loop-simplify -loop-rotate -lcssa -indvars`.

A more advanced treatment of loops is currently under development on branch `lipschitz`.
It needs the Boost Interval Arithmetic library (header only) and the GiNaC library for symbolic computations.

### Debugging Info

TAFFO-EP emits several debugging messages useful for finding out where the largest error increases come from.
When `-debug-only=errorprop` is specified on `opt`'s command line, messages of the following categories are emitted:
- the absolute error computed for each instruction;
- instructions whose error could not be propagated because of their type (mostly `br` instructions) or because of missing range or error data;
- whether the initial error specified with metadata has been used for an instruction;
- whether a `cmp` instruction may yield to a wrong comparison because of the absolute error of its operand;
- whether a fixed point instruction may cause an overflow according to the range and bit width specified in metadata;
- the number of times a loop has been unrolled, or whether loop unrolling failed for that loop (tip: use `-debug-only=loop-unroll` to know why a loop could not be unrolled);
- for each `struct`, the maximum error computed for each field;
- the maximum relative error computed for each target variable.
