/**
 * correlation.h: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#ifndef CORRELATION_H
#define CORRELATION_H

/* Default to STANDARD_DATASET. */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#define STANDARD_DATASET
#endif

/* Do not define anything if the user manually defines the size. */
#if !defined(M) && !defined(N)
/* Define the possible dataset sizes. */
#ifdef MINI_DATASET
#define M 512
#define N 512
#endif

#ifdef SMALL_DATASET
#define M 1024
#define N 1024
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
#define M 2048
#define N 2048
#endif

#ifdef LARGE_DATASET
#define M 4096
#define N 4096
#endif

#ifdef EXTRALARGE_DATASET
#define M 8192
#define N 8192
#endif
#endif /* !N */

#define _PB_M POLYBENCH_LOOP_BOUND(M, m)
#define _PB_N POLYBENCH_LOOP_BOUND(N, n)

#ifndef DATA_TYPE
#define DATA_TYPE float
#define DATA_PRINTF_MODIFIER "%e "
#endif

/* Thread block dimensions for kernel 1*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_1_X 256
#define DIM_LOCAL_WORK_GROUP_KERNEL_1_Y 1

/* Thread block dimensions for kernel 2*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_2_X 256
#define DIM_LOCAL_WORK_GROUP_KERNEL_2_Y 1

/* Thread block dimensions for kernel 3*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_3_X 32
#define DIM_LOCAL_WORK_GROUP_KERNEL_3_Y 8

/* Thread block dimensions for kernel 4*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_4_X 256
#define DIM_LOCAL_WORK_GROUP_KERNEL_4_Y 1

#endif /* !CORRELATION*/
