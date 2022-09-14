/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* mvt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "mvt.h"

#ifdef _LAMP
float POLYBENCH_1D(x1_float, N, n);
float POLYBENCH_1D(x2_float, N, n);
#endif


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i __attribute__((annotate("scalar(range(0, " PB_XSTR(N) "))")));
  int j __attribute__((annotate("scalar(range(0, " PB_XSTR(N) "))")));

  for (i = 0; i < n; i++)
    {
      x1[i] = (DATA_TYPE) (i % n) / n;
      x2[i] = (DATA_TYPE) ((i + 1) % n) / n;
      y_1[i] = (DATA_TYPE) ((i + 3) % n) / n;
      y_2[i] = (DATA_TYPE) ((i + 4) % n) / n;
      for (j = 0; j < n; j++)
	A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


#ifndef _LAMP
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x1,N,n),
		 DATA_TYPE POLYBENCH_1D(x2,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x1");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x1[i]);
  }
  POLYBENCH_DUMP_END("x1");

  POLYBENCH_DUMP_BEGIN("x2");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x2[i]);
  }
  POLYBENCH_DUMP_END("x2");
  POLYBENCH_DUMP_FINISH;
}
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_mvt(int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_A_MIN) "," PB_XSTR(VAR_A_MAX) ") final)"))), N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x1, DATA_TYPE __attribute__((annotate("target('x1') scalar(range(" PB_XSTR(VAR_x1_MIN) "," PB_XSTR(VAR_x1_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(x2, DATA_TYPE __attribute__((annotate("target('x2') scalar(range(" PB_XSTR(VAR_x2_MIN) "," PB_XSTR(VAR_x2_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(y_1, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_y_1_MIN) "," PB_XSTR(VAR_y_1_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(y_2, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_y_2_MIN) "," PB_XSTR(VAR_y_2_MAX) ") final)"))), N, n);


  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(x1),
	      POLYBENCH_ARRAY(x2),
	      POLYBENCH_ARRAY(y_1),
	      POLYBENCH_ARRAY(y_2),
	      POLYBENCH_ARRAY(A));

    scale_2d(N, N, POLYBENCH_ARRAY(A), SCALING_FACTOR);
    scale_1d(N, POLYBENCH_ARRAY(x1), SCALING_FACTOR);
    scale_1d(N, POLYBENCH_ARRAY(x2), SCALING_FACTOR);
    scale_1d(N, POLYBENCH_ARRAY(y_1), SCALING_FACTOR);
    scale_1d(N, POLYBENCH_ARRAY(y_2), SCALING_FACTOR);

  #ifdef COLLECT_STATS
    stats_header();
    stats_2d("A", N, N, POLYBENCH_ARRAY(A));
    stats_1d("x1", N, POLYBENCH_ARRAY(x1));
    stats_1d("x2", N, POLYBENCH_ARRAY(x2));
    stats_1d("y_1", N, POLYBENCH_ARRAY(y_1));
    stats_1d("y_2", N, POLYBENCH_ARRAY(y_2));
  #endif

#ifndef _LAMP
  /* Start timer. */
  polybench_start_instruments;
#endif

  timer_start();
  /* Run kernel. */
  kernel_mvt (n,
	      POLYBENCH_ARRAY(x1),
	      POLYBENCH_ARRAY(x2),
	      POLYBENCH_ARRAY(y_1),
	      POLYBENCH_ARRAY(y_2),
	      POLYBENCH_ARRAY(A));
  timer_stop();

#ifdef COLLECT_STATS
  stats_2d("A", N, N, POLYBENCH_ARRAY(A));
  stats_1d("x1", N, POLYBENCH_ARRAY(x1));
  stats_1d("x2", N, POLYBENCH_ARRAY(x2));
  stats_1d("y_1", N, POLYBENCH_ARRAY(y_1));
  stats_1d("y_2", N, POLYBENCH_ARRAY(y_2));
#endif

#ifndef _LAMP
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x1);
  POLYBENCH_FREE_ARRAY(x2);
  POLYBENCH_FREE_ARRAY(y_1);
  POLYBENCH_FREE_ARRAY(y_2);
#else
  for (int i = 0; i < n; i++){
    x1_float[i] = x1[i];
    x2_float[i] = x2[i];
  }
#endif

  return 0;
}
