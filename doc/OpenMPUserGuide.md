# OpenMP User Guide
This document introduces the OpenMP support in TAFFO, describing the following:
- the high-level goals of the integration
- the supported constructs and their peculiarities
- the unsupported constructs

More technical details, such as the modifications in the `Initializer` and `Conversion` passes, are available in 
[the dev guide](./OpenMPDevGuide.md).

## Introduction
TAFFO has preliminary support for [OpenMP](https://www.openmp.org) parallel constructs.

The goals of the integration are the following:
- The usage of a simple workflow to easily patch the OpenMP runtime library functions (such as `__kmpc_fork_call`) 
 during optimizations, requiring the least amount of work necessary for intermediate passes.
- Basic support for the most common OpenMP directives, such as `omp parallel for`. 

This allows TAFFO to convert the floating-point code scoped by the supported OpenMP pragmas into fixed-point.
- Preserving functional behavior of the application with not supported directives, such as `omp task`, to avoid failures.
- Evaluate the results with benchmarks adapted to OpenMP and containing floating-points.

Note that a thorough analysis of the effects of data-dependency in the parallel execution is out of the scope of the integration.

No additional metadata is needed to integrate OpenMP code with TAFFO.

## Supported constructs
The parallel constructs below are tested and TAFFO correctly converts floating-point logic into fixed-point.

Note that although the VRA can correctly analyse parallel code region, it will not be able to successfully deduct the range of the loop.
The problem can be fixed by:
- setting manually a final range in the annotations: `scalar(range(x, y) final)` is enough to set a correct range.
 Check out additional details in in the [Annotation Syntax documentation](./AnnotationSyntax.md#data-attributes).
- setting the trip-count for the VRA directly in the command line, with `-Xvra -unroll`. 
 Check out additional details in the [Command Line Reference](./CommandLineReference.md#-unroll-count).

- `omp parallel`
- `omp for`
- `omp section`
- `omp critical`

Note that shared variables are fully supported for the above constructs.

## Unsupported constructs
Some constructs are not currently supported in TAFFO.
Note that the compiler will be conservative and no breaking optimizations will be applied.

In particular, the usage of the following constructs stops the conversion of the shared OpenMP variables from 
floating-points to fixed-points, both outside and inside the parallel code regions.

- `omp reduce`
- `omp task`
