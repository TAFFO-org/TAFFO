/**
 * syrk.h: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#ifndef SYRK_H
#define SYRK_H

/* Default to STANDARD_DATASET. */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#define STANDARD_DATASET
#endif

/* Do not define anything if the user manually defines the size. */
#if !defined(NI) && !defined(NJ)
/* Define the possible dataset sizes. */
#ifdef MINI_DATASET
#define NI 256
#define NJ 256
#endif

#ifdef SMALL_DATASET
#define NI 512
#define NJ 512
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
#define NI 1024
#define NJ 1024
#endif

#ifdef LARGE_DATASET
#define NI 2048
#define NJ 2048
#endif

#ifdef EXTRALARGE_DATASET
#define NI 4096
#define NJ 4096
#endif
#endif /* !N */

#define _PB_NI POLYBENCH_LOOP_BOUND(NI, ni)
#define _PB_NJ POLYBENCH_LOOP_BOUND(NJ, nj)

#ifndef DATA_TYPE
#define DATA_TYPE float
#define DATA_PRINTF_MODIFIER "%0.8lf "
#endif

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#endif /* !SYRK*/
