/**
 * jacobi2D.h: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#ifndef JACOBI2D_H
#define JACOBI2D_H

/* Default to STANDARD_DATASET. */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#define STANDARD_DATASET
#endif

/* Do not define anything if the user manually defines the size. */
#if !defined(TSTEPS) && !defined(N)
/* Define the possible dataset sizes. */
#ifdef MINI_DATASET
#define TSTEPS 20
#define N 256
#endif

#ifdef SMALL_DATASET
#define TSTEPS 20
#define N 512
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
#define TSTEPS 20
#define N 1024
#endif

#ifdef LARGE_DATASET
#define TSTEPS 20
#define N 2048
#endif

#ifdef EXTRALARGE_DATASET
#define TSTEPS 20
#define N 4096
#endif
#endif /* !N */

#define _PB_TSTEPS POLYBENCH_LOOP_BOUND(TSTEPS, tsteps)
#define _PB_N POLYBENCH_LOOP_BOUND(N, n)

#ifndef DATA_TYPE
#define DATA_TYPE float
#define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#endif /* !JACOBI2D*/
