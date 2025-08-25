/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* floyd-warshall.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "floyd-warshall.h"

/* Array initialization. */
static void init_array(int n, DATA_TYPE POLYBENCH_2D(path, N, N, n, n)) {
  int i __attribute__((annotate("scalar(range(-" PB_XSTR(N) ", " PB_XSTR(N) "))")));
  int j __attribute__((annotate("scalar(range(-" PB_XSTR(N) ", " PB_XSTR(N) "))")));

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      path[i][j] = i * j % 7 + 1;
      if ((i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0)
        path[i][j] = 999;
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, DATA_TYPE POLYBENCH_2D(path, N, N, n, n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("path");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, path[i][j]);
    }
  POLYBENCH_DUMP_END("path");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_floyd_warshall(int n, DATA_TYPE POLYBENCH_2D(path, N, N, n, n)) {
  int i, j, k;

#pragma scop
  for (k = 0; k < _PB_N; k++) {
    for (i = 0; i < _PB_N; i++)
      for (j = 0; j < _PB_N; j++) {
        DATA_TYPE __attribute__((annotate("scalar()"))) tmpa = path[i][k];
        DATA_TYPE __attribute__((annotate("scalar()"))) tmpb = path[k][j];
        int cond = path[i][j] < path[i][k] + path[k][j];
        if (cond)
          path[i][j] = path[i][j];
        else
          path[i][j] = tmpa + tmpb;
      }
  }
#pragma endscop
}

int main(int argc, char** argv) {
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(path, DATA_TYPE __attribute__((annotate("target('path') scalar()"))), N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(path));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_floyd_warshall(n, POLYBENCH_ARRAY(path));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(path)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(path);

  return 0;
}
