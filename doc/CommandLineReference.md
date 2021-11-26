# TAFFO Command Line Reference

**Usage:** taffo [options] file...

The specified files can be any C or C++ source code files.
Apart from the TAFFO-specific options listed below, all CLANG options are
also accepted.

## Basic Options

##### -o \<file\>             
Write compilation output to \<file\>

#### -O\<level\>
Set the optimization level to the specified value.
The accepted optimization levels are the same as `clang`.
(-O, -O1, -O2, -O3, -Os, -Of)
                        
#### -c
Do not link the output

#### -S                    
Produce an assembly file in output instead of a binary
(overrides -c and -emit-llvm)

#### -emit-llvm
Produce a LLVM-IR assembly file in output instead of
a binary (overrides -c and -S)

#### -enable-err
Enable the error propagator (disabled by default)

#### -err-out \<file\>
Produce a textual report about the estimates performed
by the Error Propagator in the specified file.

#### -float-output \<file\>
Also compile the files without using TAFFO and store
the output to the specified location.

#### -Xinit \<option\>
Pass the specified option to the Initializer pass of
TAFFO

#### -Xvra \<option\>
Pass the specified option to the VRA pass of TAFFO

#### -Xdta \<option\>
Pass the specified option to the DTA pass of TAFFO

#### -Xconversion \<option\>
Pass the specified option to the Conversion pass of
TAFFO

#### -Xerr \<option\>
Pass the specified option to the Error Propagator pass
of TAFFO

## Pass-Specific Options

These options must be specified using the `-X(init|vra|dta|conversion|err)`
basic options. If the pass-specific option has an argument, it must be
preceded by another `-X(init|vra|dta|conversion|err)` specifier.

### Initializer (init)

#### -manualclone
Enables function cloning only for annotated functions.

### Value Range Analysis (vra)

#### -propagate-all
Propagate ranges for all functions, not only those marked as starting point.

#### -unroll \<count\>
Default loop unroll count. Setting this to 0 disables loop unrolling. 
(Default: 1)

#### -max-unroll \<count\>
Max loop unroll count. Setting this to 0 disables loop unrolling. (Default: 256)

### Data Type Allocation (dta)

#### -minfractbits \<bits\>
Minimum number of significant bits allowed in fixed point types (default: 3)

#### -totalbits \<bits\>
Minimum size of fixed point types. If the type size causes the number of
significant bits to be lower than the minimum, the size is automatically
increased. (default: 32)

#### -similarbits \<bits\>
Maximum difference in fractional part size allowed when considering a merge
between two fixed point types. (default: 2)

#### -notypemerge
Disables similar adjacent type merging optimization.

### Conversion (conversion)

(no options)

### Error Propagation (err)

#### -unroll \<count\>
Default loop unroll count. (Default: 1)

#### -max-unroll \<count\>
Max loop unroll count. Setting this to 0 disables loop unrolling. (Default: 256)

#### -cmpthresh \<perc\>
CMP errors are signaled only if error is above \<perc\> %. (default: 0)

#### -recur \<count\>
Default number of recursive calls to the same function. (default: 0)

#### -startonly
Propagate only functions with start metadata.

#### -relerror
Output relative errors instead of absolute errors (experimental).

#### -exactconst
Treat all constants as exact.

#### -sloppyaa
Enable sloppy Alias Analysis, for when LLVM AA fails.

## Experimental Options

**Warning:** These options may not work correctly in the current release of
TAFFO.

#### -feedback
Enable the feedback cycle using the Performance
Estimator and the Error Propagator.

#### -pe-model \<file\>
Uses the specified file as the performance model for
the Performance Estimator. Performance models can be
produced using the taffo-pe-train-* tools.

## Debugging Options

#### -disable-vra
Disables the VRA analysis pass, and replaces it with
a simpler, optimistic, and potentially incorrect greedy
algorithm.

#### -no-mem2reg
Disable scheduling of the mem2reg pass.

#### -debug
Enable LLVM and TAFFO debug logging during the
compilation.

#### -debug-taffo
Enable TAFFO-only debug logging during the compilation.

#### -temp-dir \<dir\>
Store various temporary files related to the execution
of TAFFO to the specified directory.
