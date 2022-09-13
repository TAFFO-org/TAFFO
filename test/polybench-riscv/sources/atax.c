/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* atax.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "atax.h"

#ifdef _LAMP
float POLYBENCH_1D(y_float, N, n);
#endif


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(tmp,M,m)
               )
{
  int i __attribute__((annotate("scalar(range(0, " PB_XSTR(N) ") final)")));
  int j __attribute__((annotate("scalar(range(0, " PB_XSTR(N) ") final)")));
  DATA_TYPE __attribute__((annotate("scalar()"))) fn;
  fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++) {
    x[i] = 1 + (i / fn);
    y[i] = 0;
  }
  for (i = 0; i < m; i++) {
    tmp[i] = 0;
    for (j = 0; j < n; j++) {
      A[i][j] = (DATA_TYPE)((i + j) % n) / (5 * m);
    }
  }
}


#ifndef _LAMP
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_atax(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    y[i] = 0;
  for (i = 0; i < _PB_M; i++)
    {
      tmp[i] = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_N; j++)
	tmp[i] = tmp[i] + A[i][j] * x[j];
      for (j = 0; j < _PB_N; j++)
	y[j] = y[j] + A[i][j] * tmp[i];
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_A_MIN) "," PB_XSTR(VAR_A_MAX) ") final)"))), M, N, m, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_x_MIN) "," PB_XSTR(VAR_x_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE __attribute__((annotate("target('y') scalar(range(" PB_XSTR(VAR_y_MIN) "," PB_XSTR(VAR_y_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_tmp_MIN) "," PB_XSTR(VAR_tmp_MAX) ") final)"))), M, m);

  /* Initialize array(s). */
  init_array (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(tmp));

  scale_2d(M, N, POLYBENCH_ARRAY(A), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(x), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(y), SCALING_FACTOR);
  scale_1d(M, POLYBENCH_ARRAY(tmp), SCALING_FACTOR);


#ifdef COLLECT_STATS
  stats_header();
  stats_2d("A", M, N, POLYBENCH_ARRAY(A));
  stats_1d("x", N, POLYBENCH_ARRAY(x));
  stats_1d("y", N, POLYBENCH_ARRAY(y));
  stats_1d("tmp", M, POLYBENCH_ARRAY(tmp));
#endif

#ifndef _LAMP
  /* Start timer. */
  polybench_start_instruments;
#endif

  timer_start();
  /* Run kernel. */
  kernel_atax (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(x),
	       POLYBENCH_ARRAY(y),
	       POLYBENCH_ARRAY(tmp));

  timer_stop();

#ifdef COLLECT_STATS
  stats_2d("A", M, N, POLYBENCH_ARRAY(A));
  stats_1d("x", N, POLYBENCH_ARRAY(x));
  stats_1d("y", N, POLYBENCH_ARRAY(y));
  stats_1d("tmp", M, POLYBENCH_ARRAY(tmp));
#endif

#ifndef _LAMP
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);
#else
  for (int i = 0; i < N; i++)
    y_float[i] = y[i];
#endif

  return 0;
}
