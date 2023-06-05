# TAFFO Simple Test Cases

A suite of regression tests for TAFFO


## Usage

TAFFO must have been compiled and installed, and its binaries must be in the current `$PATH`.

First, do
```bash
$ export LLVM_DIR=/path/to/LLVM/install
```

To run them all, just do
```bash
$ ./run-test.sh
```

To run only one of them, do
```bash
$ ./run-test.sh --only testcase.c
```

To delete all build files produced by the tests, run
```bash
$ ./run-test.sh clean
```
