# CLoverLeaf_Serial

A C port of the serial (no-MPI) version of CloverLeaf, for use with [TAFFO](https://github.com/TAFFO-org/TAFFO) (Tuning Assistant for Floating Point to Fixed Point Optimization).

Test is based on: [lynxnb/CLoverLeaf_Serial](https://github.com/lynxnb/CLoverLeaf_Serial)  (see the `embedded` branch)  
Original repository: [UK-MAC/CloverLeaf_Serial](https://github.com/UK-MAC/CloverLeaf_Serial)

## Performance

Performance should be on par with the Fortran version using C kernels.

## Building

Building is done using GNU Make. See the Makefile available targets and options.

## TAFFO Usage

To ease the process of annotating the code, an effort was made to be able to generate annotations automatically.
A rudimental usage tracker is able to periodically collect data about the program's working arrays, thus being able to report the minimum and maximum values used per-array.
It can then print out those values as a human-readable report, or as ready-to-use TAFFO annotations, which can then be applied to the code directly.

To enable the usage tracker, user callbacks need to be enabled at compile time:
```bash
make USER_CALLBACKS=1
```
Usage info will be saved to the `usage_tracker.txt` file.

## User Callbacks
User callbacks are functions that can be used to run custom code at specific execution points in the program, allowing for user defined code to be executed without having to modify the original source. For example, they are used to run the usage tracker.

## Building for embedded

Make targets to compile as static library are available, both with GCC and TAFFO.

Running CloverLeaf on a STM board is as simple as getting miosix to run, and then calling the `clover_main()` entrypoint. To change the input values used by CloverLeaf, the `initialise.c` file must be edited as there's usually no filesystem on embedded boards.
```bash
make clover_leaf.a ARMv7=1
```

See the `InputDecks` folder for valid input values, although the only ones that will likely run on a smaller board are the `small`, `smaller` and `sod[x|y|xy]` decks. The default hardcoded input values (`smaller` input deck) take up ~120kb of heap.

The `main.cpp` file is provided as an example of what the miosix entrypoint should look like. To run CloverLeaf, clone Miosix v2 in this folder, adjust Miosix config values for the target board, then generate the desired static library version of CloverLeaf (GCC or TAFFO) and link it with Miosix and the main entrypoint.
