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
/* Default data type is double, default size is 20x1000. */
#include "jacobi-2d-imper.h"

/* Array initialization. */
static void init_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n)) {
  int i __attribute__((annotate("scalar(range(0, " PB_XSTR(N) ") final)")));
  int j __attribute__((annotate("scalar(range(0, " PB_XSTR(N) ") final)")));

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = ((DATA_TYPE) i * (j + 2) + 2) / n;
      B[i][j] = ((DATA_TYPE) i * (j + 3) + 3) / n;
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * n + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_jacobi_2d_imper(int tsteps,
                                   int n,
                                   DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                                   DATA_TYPE POLYBENCH_2D(B, N, N, n, n)) {
  int t, i, j;

#pragma scop

#pragma omp parallel private(i, j, t)
  {
#pragma omp master
    {
      for (t = 0; t < _PB_TSTEPS; t++) {
#pragma omp parallel for schedule(static)
        for (i = 1; i < _PB_N - 1; i++)
          for (j = 1; j < _PB_N - 1; j++)
            B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]);
#pragma omp parallel for schedule(static)
        for (i = 1; i < _PB_N - 1; i++)
          for (j = 1; j < _PB_N - 1; j++)
            A[i][j] = B[i][j];
      }
    }
  }
#pragma endscop
}

int main(int argc, char** argv) {
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(
    A, DATA_TYPE __attribute__((annotate("target('A') scalar(range(0,1000) final)"))), N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(
    B, DATA_TYPE __attribute__((annotate("target('A') scalar(range(0,1000) final)"))), N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_2d_imper(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
