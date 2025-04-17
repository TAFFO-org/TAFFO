/**
 * fdtd2d.h: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#ifndef FDTD2D_H
#define FDTD2D_H

/* Default to STANDARD_DATASET. */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#define STANDARD_DATASET
#endif

/* Do not define anything if the user manually defines the size. */
#if !defined(TMAX) && !defined(NX) && !defined(NY)
/* Define the possible dataset sizes. */
#ifdef MINI_DATASET
#define TMAX 500
#define NX 512
#define NY 512
#endif

#ifdef SMALL_DATASET
#define TMAX 500
#define NX 1024
#define NY 1024
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
#define TMAX 500
#define NX 2048
#define NY 2048
#endif

#ifdef LARGE_DATASET
#define TMAX 500
#define NX 4096
#define NY 4096
#endif

#ifdef EXTRALARGE_DATASET
#define TMAX 500
#define NX 8192
#define NY 8192
#endif
#endif /* !N */

#define _PB_TMAX POLYBENCH_LOOP_BOUND(TMAX, tmax)
#define _PB_NX POLYBENCH_LOOP_BOUND(NX, nx)
#define _PB_NY POLYBENCH_LOOP_BOUND(NY, ny)

#ifndef DATA_TYPE
#define DATA_TYPE float
#define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#endif /* !FDTD2D*/
