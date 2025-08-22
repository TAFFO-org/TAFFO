/* POLYBENCH/GPU-OPENMP
 *
 * This file is a part of the Polybench/GPU-OpenMP suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 *
 * Copyright 2013, The University of Delaware
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 10x1024x1024. */
#include "adi.h"

/* Array initialization. */
static void init_array(int n,
                       DATA_TYPE POLYBENCH_2D(X, N, N, n, n),
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                       DATA_TYPE POLYBENCH_2D(B, N, N, n, n)) {
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      X[i][j] = ((DATA_TYPE) i * (j + 1) + 1) / n;
      A[i][j] = ((DATA_TYPE) i * (j + 2) + 2) / n;
      B[i][j] = ((DATA_TYPE) i * (j + 3) + 3) / n;
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, DATA_TYPE POLYBENCH_2D(X, N, N, n, n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, X[i][j]);
      if ((i * N + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_adi(int tsteps,
                       int n,
                       DATA_TYPE POLYBENCH_2D(X, N, N, n, n),
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                       DATA_TYPE POLYBENCH_2D(B, N, N, n, n)) {
  int t, i1, i2;

#pragma omp parallel private(t, i1, i2)
  {
#pragma omp master
    {
      for (t = 0; t < _PB_TSTEPS; t++) {
#pragma omp for
        for (i1 = 0; i1 < _PB_N; i1++)
          for (i2 = 1; i2 < _PB_N; i2++) {
            X[i1][i2] = X[i1][i2] - X[i1][i2 - 1] * A[i1][i2] / B[i1][i2 - 1];
            B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2 - 1];
          }
#pragma omp for
        for (i1 = 0; i1 < _PB_N; i1++)
          X[i1][_PB_N - 1] = X[i1][_PB_N - 1] / B[i1][_PB_N - 1];
#pragma omp for
        for (i1 = 0; i1 < _PB_N; i1++)
          for (i2 = 0; i2 < _PB_N - 2; i2++)
            X[i1][_PB_N - i2 - 2] =
              (X[i1][_PB_N - 2 - i2] - X[i1][_PB_N - 2 - i2 - 1] * A[i1][_PB_N - i2 - 3]) / B[i1][_PB_N - 3 - i2];
#pragma omp for
        for (i1 = 1; i1 < _PB_N; i1++)
          for (i2 = 0; i2 < _PB_N; i2++) {
            X[i1][i2] = X[i1][i2] - X[i1 - 1][i2] * A[i1][i2] / B[i1 - 1][i2];
            B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1 - 1][i2];
          }
#pragma omp for
        for (i2 = 0; i2 < _PB_N; i2++)
          X[_PB_N - 1][i2] = X[_PB_N - 1][i2] / B[_PB_N - 1][i2];
#pragma omp for
        for (i1 = 0; i1 < _PB_N - 2; i1++)
          for (i2 = 0; i2 < _PB_N; i2++)
            X[_PB_N - 2 - i1][i2] =
              (X[_PB_N - 2 - i1][i2] - X[_PB_N - i1 - 3][i2] * A[_PB_N - 3 - i1][i2]) / B[_PB_N - 2 - i1][i2];
      }
    }
  }
}

int main(int argc, char** argv) {
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(X, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_adi(tsteps, n, POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(X)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(X);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
