<img src="doc/logo/TAFFO-logo-black.png" alt="TAFFO" width=30%>

*(Tuning Assistant for Floating Point to Fixed Point Optimization)*

TAFFO is an autotuning framework which tries to replace floating point operations with fixed point operations as much as possible.

It is based on LLVM 15 and has been tested on Linux (any attempt to compile on Windows, WSL or macOS is at your own risk and peril).

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

At the moment TAFFO supports LLVM 14 and 15. No other version is supported.
Moreover, LLVM plugins compiled for a given major version of LLVM cannot be loaded by any other version. Therefore it is not a good idea to redistribute TAFFO as a binary.
If you are building LLVM from sources, you must configure it with `-DLLVM_BUILD_LLVM_DYLIB=ON` and `-DLLVM_LINK_LLVM_DYLIB=ON` for the TAFFO build to succeed.

TAFFO requires a single additional dependency: Google ORTools.
It is possible to build and install ORTools manually, or you can build it as part of TAFFO by specifying `-DTAFFO_BUILD_ORTOOLS=ON`.
This option is recommended.

```sh
cd /path/to/the/location/of/TAFFO
export LLVM_DIR=/usr/lib/llvm-15 # optional
mkdir build
cd build
cmake .. -DTAFFO_BUILD_ORTOOLS=ON
cmake --build .
cmake --build . --target install
```

If you want to modify TAFFO or see the debugging logs you need to also build LLVM in debug mode first.
You are encouraged to follow our guide: [doc/BuildingLLVM.md](Building LLVM)

### 2: Modify and test the application

Modify the application to insert annotations on the appropriate variable declarations, then use `taffo` to compile your application.

```sh
<editor> program.c
[...]
taffo -O3 -o program-taffo program.c
```

See the annotation syntax documentation or the examples in `test/simple-test-cases` to get an idea on how to write annotations. You can also test TAFFO without adding annotations, which will produce the same results as using `clang` as a compiler/linker instead of `taffo`.

Note that there is no `taffo++`; C++ source files are autodetected by the file extension instead.

## How to build and run unit tests and integration tests

Unit tests are located in `unittests` directory, while
integration tests and benchmarks are located in `test` directory.

```shell
mkdir build
cd build/
cmake .. -DTAFFO_BUILD_ORTOOLS=ON -DTAFFO_UNITTESTS=ON
cmake --build .
ctest -VV
```

If `-DTAFFO_UNITTESTS=ON` is not specified, only the integration tests will be run.

For further information about the unit tests, check the dedicated [unittests/README.md](unittests/README.md) file.
and the `readme`s in the integration test directory. 

**Notice:** Some integration tests do not work in the current version of TAFFO.
This is a known issue and will be fixed in a later revision.
