# AxBench TAFFO Benchmarks

http://axbench.org/

TAFFO must have been compiled and installed, and its binaries must be in the
current `$PATH`.

Additional dependencies to install for computing the error correctly:

 - Python3, pypng
 - Python2
 - ImageMagick

To try out the benchmarks you can use both the top-level Makefile and the single
makefiles in each benchmark's directory. All targets are valid for both kinds
of makefiles.

The available targets are:

 - `all`: builds the benchmark(s) with and without TAFFO
 - `run`: runs the benchmark(s)
 - `validate`: prints a table comparing the execution time of the unmodified 
   version and the TAFFO version, and the error of the outputs of the TAFFO
   version compared with the unmodified version.
 - `clean`: cleans up intermediate files (executables, data output, logs)

In order to run all the benchmarks, execute the following commands in this
directory:

```bash
$ make clean
$ make
$ make run
$ make validate
```
