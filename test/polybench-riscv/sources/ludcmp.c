/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* ludcmp.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "ludcmp.h"

#ifdef _LAMP
float POLYBENCH_1D(x_float, N, n);
#endif


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i __attribute__((annotate("scalar(range(0, " PB_XSTR(N) ") final)")));
  int j __attribute__((annotate("scalar(range(0, " PB_XSTR(N) "))")));
  DATA_TYPE __attribute__((annotate("scalar()"))) fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      x[i] = 0;
      y[i] = 0;
      b[i] = (i+1)/fn/2.0 + 4;
    }

  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite. */
  /* not necessary for LU, but using same code as cholesky */
  int r,s,t;
#ifndef _LAMP
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE __attribute__((annotate("scalar()"))), N, N, n, n);
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      (POLYBENCH_ARRAY(B))[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	(POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
  POLYBENCH_FREE_ARRAY(B);
#else
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      B[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	      B[r][s] += A[r][t] * A[s][t];
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	A[r][s] = B[r][s];
#endif
}


#ifndef _LAMP
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_ludcmp(int n,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(b,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j, k;

  DATA_TYPE __attribute__((annotate("scalar(range(-2, 2) final)"))) w;

#pragma scop
  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j <i; j++) {
       w = A[i][j];
       for (k = 0; k < j; k++) {
          w -= A[i][k] * A[k][j];
       }
        A[i][j] = w / A[j][j];
    }
   for (j = i; j < _PB_N; j++) {
       w = A[i][j];
       for (k = 0; k < i; k++) {
          w -= A[i][k] * A[k][j];
       }
       A[i][j] = w;
    }
  }

  for (i = 0; i < _PB_N; i++) {
     w = b[i];
     for (j = 0; j < i; j++)
        w -= A[i][j] * y[j];
     y[i] = w;
  }

   for (i = _PB_N-1; i >=0; i--) {
     w = y[i];
     for (j = i+1; j < _PB_N; j++)
        w -= A[i][j] * x[j];
     x[i] = w / A[i][i];
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_A_MIN) "," PB_XSTR(VAR_A_MAX) ") final)"))), N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_B_MIN) "," PB_XSTR(VAR_B_MAX) ") final)"))), N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_b_MIN) "," PB_XSTR(VAR_b_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE __attribute__((annotate("target('x') scalar(range(" PB_XSTR(VAR_x_MIN) "," PB_XSTR(VAR_x_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_y_MIN) "," PB_XSTR(VAR_y_MAX) ") final)"))), N, n);


  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(b),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y));

  scale_2d(N, N, POLYBENCH_ARRAY(A), SCALING_FACTOR);
  scale_2d(N, N, POLYBENCH_ARRAY(B), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(b), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(x), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(y), SCALING_FACTOR);

#ifdef COLLECT_STATS
  stats_header();
  stats_2d("A", N, N, POLYBENCH_ARRAY(A));
  stats_2d("B", N, N, POLYBENCH_ARRAY(B));
  stats_1d("b", N, POLYBENCH_ARRAY(b));
  stats_1d("x", N, POLYBENCH_ARRAY(x));
  stats_1d("y", N, POLYBENCH_ARRAY(y));
#endif

#ifndef _LAMP
  /* Start timer. */
  polybench_start_instruments;
#endif

  timer_start();
  /* Run kernel. */
  kernel_ludcmp (n,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(b),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y));
  timer_stop();

#ifdef COLLECT_STATS
  stats_2d("A", N, N, POLYBENCH_ARRAY(A));
  stats_2d("B", N, N, POLYBENCH_ARRAY(B));
  stats_1d("b", N, POLYBENCH_ARRAY(b));
  stats_1d("x", N, POLYBENCH_ARRAY(x));
  stats_1d("y", N, POLYBENCH_ARRAY(y));
#endif

#ifndef _LAMP
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
#else
  for (int i = 0; i < n; i++)
    x_float[i] = x[i];
#endif

  return 0;
}
