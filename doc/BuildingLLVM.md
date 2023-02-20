# Building LLVM for debugging TAFFO

When built in release mode, LLVM replaces all debug prints with no-operations.
Therefore it's hard or impossible to debug TAFFO, unless LLVM is built in
debug mode. However, building LLVM is notoriously resource-intensive and
in addition to that, debug-mode LLVM builds typically take a lot of disk space
(because of the symbols).

This document describes the recommended procedure for building LLVM correctly
for use with TAFFO in Debug mode without using too much disk space. It assumes
the use of a Unix-like operating system like Linux or macOS. Windows users
can use WSL, following the instructions for Linux.

Make sure, before you start, that you have at least 80GB of disk space
(and some spare just in case). If you are on Linux, also please enable swap
before the OOM killer reaps the wrong process.

### 0: Install requirements

First of all, **you** need to have your brain turned on and working. This is
not obvious to many. Prior experience in the software witchcraft thingamajigs
is also very useful as this is not a guide written for complete beginners
(otherwise it would be 10 times as long and it would be useless for everybody,
while now it is only useless for somebody).

LLVM does not require particular dependencies, except for `cmake`, a
cross-platform build system, so make sure you have that available together with
working C and C++ compilers (GCC 9+ and clang 10+ should work). Also install
`ninja` (often packaged as `ninja-build`) which we will use together with
`cmake`.

On **Linux** only, also install `gold`, an alternative linker. This will speed
up the build process.

Additionally this guide uses `wget`, but if you don't have it you can download
files through your browser or using `curl`.

### 1: Download LLVM from the official tarball

The tarball takes less space than the git repository but no tarballs are
released for unstable versions of LLVM. However TAFFO never requires unstable
versions of LLVM, so we can use the tarball without worry.

```
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.7/llvm-project-15.0.7.src.tar.xz
tar xvf llvm-project-15.0.7.src.tar.xz
```

These commands will expand the entire LLVM project source code into the
directory `llvm-project-15.0.7.src`.

### 2: Configure LLVM with CMake

Create a temporary build directory and move into it:

```
mkdir build
cd build
```

Now comes the time to configure LLVM, and actually specify the compilation
options. Here is the suggested command.

**Do not run this command yet, read on!**

```
cmake ../llvm \
  -GNinja \
  -DCMAKE_INSTALL_PREFIX=/opt/llvm-15-d \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_PROJECTS='clang;compiler-rt;libcxx;libcxxabi;openmp' \
  -DLLVM_ENABLE_LIBCXX=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_TARGETS_TO_BUILD='X86;ARM;AArch64' \
```

Additional options to add at the end **only for Linux**:

```
  -DLLVM_USE_LINKER=gold \
  -DLLVM_PARALLEL_LINK_JOBS=1
```

Additional options to add at the end **only for macOS**:

```
  -DCMAKE_IGNORE_PATH='/opt/local/bin;/opt/local/include;/opt/local/lib;/usr/local/bin;/usr/local/include;/usr/local/lib' \
  -DLLVM_EXTERNALIZE_DEBUGINFO=ON
```

**Now you can run the cmake command. Make sure to copy-paste both sections
(the common one and the OS-specific one which matches your OS).**

There is an appendix to this document which explains the rationale behind all
the options, make sure to check it out if you are considering the removal of
any of them.

In general we recommend keeping them all, as they are time-tested and
known working.

### 3: Building LLVM

Just run

```
ninja
```

Then you can make some tea, or coffee, or both, or even have lunch **and**
dinner probably.

We do not advise looking at the progress indicator continuously as that might
lead to starvation and undesirable obsessive behavior.

### 4: Installing LLVM

Assuming you are still alive, now it's the time to install LLVM.

Just run:

```
sudo ninja install
```

All LLVM files will be copied in the directory specified in the
`CMAKE_INSTALL_PREFIX` option.
This is also the same directory you will specify in the `LLVM_DIR` option when
building TAFFO (see the top-level Readme for more information).

You can check that everything works by running:

```
/opt/llvm-15-d/bin/clang --version
```

**macOS only:**

It is desirable to also move all the .dSYM files out of the build folder,
otherwise when you delete it you will be left without debugging symbols.

This can be done with the following one-liner script:

```
out=/opt/llvm-15-d; for fn in $(find . -name '*.dSYM'); do base=$(basename "$fn" .dSYM); n=$(find $out -path "$out/src" -prune -o -name "$base" -print); if [[ -e "$n" ]]; then n=$(dirname "$n"); else sudo mkdir -p $out/dSYM; n=$out/dSYM; fi; echo "$fn" '->' "$n"; sudo cp -R -c "$fn" "$n"; done
```

Whatever the operating system, at this point you can delete the `build`
folder (which will take quite the amount of disk space).

And now we are done!

### Additional notes regarding macOS

If you want to use the `clang` and `clang++` build you made to compile some
program, you must prepend `xcrun` to the invocation, otherwise none of the
system headers will be found. This is macOS's solution to being able to
swap the system headers depending on the SDK you want to use (professional
software developers often prefer to use a newer/older SDK than the one of
the current system for backwards compatibility, and this is the solution for
accomodating that).

Example:

```
xcrun /opt/llvm-15-d/bin/clang -o xxx yyy.c
```

## References

More info about building LLVM can be found here:
[https://llvm.org/docs/CMake.html](https://llvm.org/docs/CMake.html)

Remember that StackOverflow and Google are your best bet if something does not
work.

## Appendix: explainer for all the CMake options

The following options are mandatory:

- `-DCMAKE_BUILD_TYPE=Debug`: Specifies that we want a Debug-mode build
- `-DLLVM_ENABLE_PROJECTS=...`: Specifies which sub-projects of LLVM we want to
  build.
  
  `clang` is used as the C and C++ frontend by TAFFO.
  
  `compiler-rt` is a library of functions required for properly linking any
  clang-compiled program (the equivalent of libgcc if you are familiar with
  GCC).
  
  `libcxx;libcxxabi` are required for C++ support in clang to work, otherwise
  STL headers will not be in the search paths and linking with the C++ runtime
  will fail.
  
  `openmp` enables OpenMP support in clang and the OpenMP runtime.
- `-DLLVM_ENABLE_LIBCXX=ON`: Required with `libcxx;libcxxabi`.
- `-DLLVM_INSTALL_UTILS=ON`: Also builds and installs various utilities that
  we use to inspect LLVM-IR code, "link" different compilation units together,
  optimize them, and so on.

These other options are recommended but can be removed/changed if one wishes:

- `-GNinja`: Produces makefiles for Ninja, a Make replacement. Ninja is faster
  than Make at doing the same job, so might as well use it.
- `-DLLVM_INCLUDE_EXAMPLES=OFF`: Disables building and installing various
  sets of example code.
- `-DLLVM_BUILD_LLVM_DYLIB=ON`: Builds LLVM as a shared library, which is
  something that provides a lot of disk space savings.
- `-DLLVM_LINK_LLVM_DYLIB=ON`: Ensures that all tools use the LLVM shared
  library instead of linking statically.
- `-DLLVM_OPTIMIZED_TABLEGEN=ON`: Builds "tablegen", a tool which is required
  for building LLVM, in Release mode rather than in Debug mode. TAFFO does
  not use this tool, thus building it in Release makes it faster, in turn
  making LLVM's compilation faster.
- `-DLLVM_TARGETS_TO_BUILD='X86;ARM;AArch64'` Specifies the set of CPU
  architectures we want to support. By default all architectures are built,
  by specifying only the ones we are interested in we shorten build times.
- `-DCMAKE_INSTALL_PREFIX=/opt/llvm-15-d`: Specifies we want to install the
  build products into a separate directory and not /usr/local. **If you install
  into /usr/local you risk using the EXTRA-SUPER-DUPER SLOW clang we build for
  NORMAL NON-TAFFO-RELATED STUFF, so remove this option AT YOUR RISK AND
  PERIL.**

The following options are OS-dependent but can be removed if you wish:

- `-DLLVM_USE_LINKER=gold`: The default linker on Linux (GNU ld) is so slow
  and uses so much RAM that building LLVM can consume more than 32 GB of RAM
  even on a 4 core machine. By specifying this option we tell CMake to use
  the `gold` linker instead, which uses less RAM and is faster.
  
  GNU ld has been observed grinding a 20 core machine with 70 GB of RAM to a
  complete halt because 20 parallel linking jobs exhausted the entire 70 GBs.
  Whatever your machine is, it is guaranteed it is not powerful enough to build
  LLVM with GNU ld.
- `-DLLVM_PARALLEL_LINK_JOBS=1`: To improve our chances of success on Linux even
  more the LLVM devs have introduced this option which limits the amount of
  parallel linking jobs. `1` is a conservative value here, on a 20 core machine
  you may increase it up to `5` (more than 5 and you go back to the swapfest
  grind-to-a-halt situation) given that you use `gold` and not GNU ld.
  
  Better keep this value to 1 for a normal desktop machine.
- `-DCMAKE_IGNORE_PATH=...`: On macOS it is normal for people to install
  third-party package manager (Homebrew, MacPorts) and other stuff via .pkg
  installers in /usr/local. This software may interfere with CMake when it
  searches for optional LLVM dependencies, and if it finds old/stale/broken
  software around your disk it might cause hard-to-debug issues.
  Therefore, better to ignore all user-installed software, which is what this
  command does.
- `-DLLVM_EXTERNALIZE_DEBUGINFO=ON`: macOS is weird because it does not
  include DWARF debug information in linked binaries. Instead, the debug info
  is kept in object files, and then the debugger uses Spotlight (!) to match
  the executable with the object files to fetch the debug info from there.
  However if we want to eventually remove all build intermediates (= the
  object files) we need to store the debug info somewhere by itself.
  This is where .dSYM bundles come into action. These bundles are directories
  which contain stub object files with no code but just the debug info.
  If these bundles are available *somewhere* on your system then they will be
  automatically picked up and used when debugging.
  
  This options enables creating the .dSYM bundles during compilation.

(note: the macOS linker does not have the performance issues that GNU ld has,
so no linker-related options are needed there.)

