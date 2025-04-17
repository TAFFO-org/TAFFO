/**
 * gesummv.h: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#ifndef GESUMMV_H
#define GESUMMV_H

/* Default to STANDARD_DATASET. */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#define STANDARD_DATASET
#endif

/* Do not define anything if the user manually defines the size. */
#if !defined(N)
/* Define the possible dataset sizes. */
#ifdef MINI_DATASET
#define N 1024
#endif

#ifdef SMALL_DATASET
#define N 2048
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
#define N 4096
#endif

#ifdef LARGE_DATASET
#define N 8192
#endif

#ifdef EXTRALARGE_DATASET
#define N 16384
#endif
#endif /* !N */

#define _PB_N POLYBENCH_LOOP_BOUND(N, n)

#ifndef DATA_TYPE
#define DATA_TYPE float
#define DATA_PRINTF_MODIFIER "%0.8lf "
#endif

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 256
#define DIM_LOCAL_WORK_GROUP_Y 1

#endif /* !GESUMMV*/
