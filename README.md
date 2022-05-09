<img src="doc/logo/TAFFO-logo-black.png" alt="TAFFO" width=30%>

*(Tuning Assistant for Floating Point to Fixed Point Optimization)*

TAFFO is an autotuning framework which tries to replace floating point operations with fixed point operations as much as possible.

It is based on LLVM 12 and has been tested on Linux (any attempt to compile on Windows, WSL or macOS is at your own risk and peril).

## How to use TAFFO

Taffo currently ships as 5 LLVM plugins, each one of them containing one LLVM optimization or analysis pass:

 - TaffoInitializer (Initialization pass)
 - TaffoVRA (Value Range Analysis pass)
 - TaffoDTA (Data Type Allocation pass)
 - LLVMFloatToFixed (Conversion pass)
 - LLVMErrorPropagator (Error Propagation pass of the Feedback Estimator)

To execute TAFFO, a simple frontend is provided named `taffo`, which can be substituted to `clang` in order to compile or link executables.
Behind the scenes, it uses the LLVM `opt` tool to load one pass at a time and run it on LLVM IR source files.

To use TAFFO it is encouraged to follow these steps:

### 1: Build and install TAFFO

Create a build directory, compile and install TAFFO.
You can either install TAFFO to the standard location of `/usr/local`, or you can install it to any other location of your choice.
In the latter case you will have to add that location to your PATH.

If you have multiple LLVM versions installed and you want to link TAFFO to a specific one, set the `LLVM_DIR` environment variable to the install prefix of the correct LLVM version beforehand.

Note that at the moment TAFFO supports only LLVM 12, and that LLVM plugins compiled for a given major version of LLVM cannot be loaded by any other version.
If you are building LLVM from sources, you must configure it with `-DLLVM_BUILD_LLVM_DYLIB=ON` and `-DLLVM_LINK_LLVM_DYLIB=ON` for the TAFFO build to succeed.

Additionally, TAFFO requires Google ORTools installed.
It is possible to build and install ORTools manually, or you can build this dependency as part of TAFFO by specifying `-DTAFFO_BUILD_ORTOOLS=ON`.
This option is recommended.

```sh
$ cd /path/to/the/location/of/TAFFO
$ export LLVM_DIR=/usr/lib/llvm-12 # optional
$ mkdir build
$ cd build
$ cmake .. -DTAFFO_BUILD_ORTOOLS=ON
$ cmake --build .
```

### 2: Modify and test the application

Modify the application to insert annotations on the appropriate variable declarations, then use `taffo` to compile your application.

```sh
<editor> program.c
[...]
taffo -O3 -o program-taffo program.c
```

See the annotation syntax documentation or the examples in `test/simple-test-cases` to get an idea on how to write annotations. You can also test TAFFO without adding annotations, which will produce the same results as using `clang` as a compiler/linker instead of `taffo`.

Note that there is no `taffo++`; C++ source files are autodetected by the file extension instead.

## How to run unit tests

Unit tests are located in `unittests` directory.

```shell
mkdir build
cd build
cmake .. -DTAFFO_BUILD_ORTOOLS=ON
cmake --build .
ctest -VV
```

## How to run integration tests

```shell
mkdir build
cd build
cmake -DTAFFO_BUILD_ORTOOLS=ON -DCMAKE_INSTALL_PREFIX=../dist -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=1 -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --target install
```

This will have your build of taffo installed in `<project_dir>/dist` directory.
Next, you need to export the taffo binaries paths to PATH.
You also need to set LLVM_DIR env variable to the PATH of your LLVM installation.
A good way to do that is to use https://github.com/direnv/direnv .
This way you can save your local configuration in a `.envrc` file 
that will be loaded automatically in the directory of the tests.
An example of `.envrc` file:

```shell
# tests/simple-test-cases/.envrc
PATH_add /<path-to-your-project-dir>/dist/bin
PATH_add /<path-to-your-project-dir>/dist/lib
export LLVM_DIR=/usr/lib/llvm-12
```

Then you can follow the instructions given in the test suite readme. 
Integration tests and benchmarks are located in `test` directory.

**Notice:** Some integration tests do not work in the current release of TAFFO 0.3.
This is a known issue and will be fixed in a later revision.
