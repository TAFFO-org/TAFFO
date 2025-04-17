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
/* Default data type is double, default size is 4000. */
#include "gemm.h"

/* Array initialization. */
static void init_array(int ni,
                       int nj,
                       int nk,
                       DATA_TYPE* alpha,
                       DATA_TYPE* beta,
                       DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
                       DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                       DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj)) {
  int i __attribute__((annotate("scalar(range(0, " PB_XSTR(NI) ") final)")));
  int j __attribute__((annotate("scalar(range(0, " PB_XSTR(NJ) ") final)")));

  *alpha = 32412;
  *beta = 2123;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = ((DATA_TYPE) i * j) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = ((DATA_TYPE) i * j) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = ((DATA_TYPE) i * j) / ni;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj, DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj)) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, C[i][j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_gemm(int ni,
                        int nj,
                        int nk,
                        DATA_TYPE alpha,
                        DATA_TYPE beta,
                        DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
                        DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                        DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj)) {
  int i, j, k;
#pragma scop
#pragma omp parallel
  {
/* C := alpha*A*B + beta*C */
#pragma omp for private(j, k)
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NJ; j++) {
        C[i][j] *= beta;
        for (k = 0; k < _PB_NK; ++k)
          C[i][j] += alpha * A[i][k] * B[k][j];
      }
  }
#pragma endscop
}

int main(int argc, char** argv) {
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha __attribute__((annotate("target('alpha') scalar()")));
  DATA_TYPE beta __attribute__((annotate("target('beta') scalar()")));
  POLYBENCH_2D_ARRAY_DECL(
    C, DATA_TYPE __attribute__((annotate("target('C') scalar(range(0, 20000000000000) final)"))), NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE __attribute__((annotate("target('A') scalar()"))), NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE __attribute__((annotate("target('B') scalar()"))), NK, NJ, nk, nj);

  /* Initialize array(s). */
  init_array(ni, nj, nk, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemm(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
