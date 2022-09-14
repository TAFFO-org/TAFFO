/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemver.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"

#ifdef _LAMP
float POLYBENCH_2D(A_float, N, N, n, n);
float POLYBENCH_1D(w_float, N, n);
float POLYBENCH_1D(x_float, N, n);
#endif


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE *alpha,
		 DATA_TYPE *beta,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(u1,N,n),
		 DATA_TYPE POLYBENCH_1D(v1,N,n),
		 DATA_TYPE POLYBENCH_1D(u2,N,n),
		 DATA_TYPE POLYBENCH_1D(v2,N,n),
		 DATA_TYPE POLYBENCH_1D(w,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i __attribute((annotate("scalar(range(0," PB_XSTR(N) ") final)")));
  int j __attribute((annotate("scalar(range(0," PB_XSTR(N) ") final)")));

  *alpha = 1.5;
  *beta = 1.2;

  DATA_TYPE __attribute((annotate("scalar()"))) fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      u1[i] = i / fn;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < n; j++)
        A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


#ifndef _LAMP
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
  }
  POLYBENCH_DUMP_END("w");
  POLYBENCH_DUMP_FINISH;
}
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemver(int n,
		   DATA_TYPE alpha,
		   DATA_TYPE beta,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(u1,N,n),
		   DATA_TYPE POLYBENCH_1D(v1,N,n),
		   DATA_TYPE POLYBENCH_1D(u2,N,n),
		   DATA_TYPE POLYBENCH_1D(v2,N,n),
		   DATA_TYPE POLYBENCH_1D(w,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n),
		   DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

#pragma scop

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < _PB_N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      w[i] = w[i] +  alpha * A[i][j] * x[j];

#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE __attribute((annotate("scalar(range(" PB_XSTR(VAR_alpha_MIN) "," PB_XSTR(VAR_alpha_MAX) ") final)"))) alpha;
  DATA_TYPE __attribute((annotate("scalar(range(" PB_XSTR(VAR_beta_MIN) "," PB_XSTR(VAR_beta_MAX) ") final)"))) beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE __attribute((annotate("scalar(range(" PB_XSTR(VAR_A_MIN) "," PB_XSTR(VAR_A_MAX) ") final)"))), N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE __attribute((annotate("scalar(range(" PB_XSTR(VAR_u1_MIN) "," PB_XSTR(VAR_u1_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE __attribute((annotate("scalar(range(" PB_XSTR(VAR_v1_MIN) "," PB_XSTR(VAR_v1_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE __attribute((annotate("scalar(range(" PB_XSTR(VAR_u2_MIN) "," PB_XSTR(VAR_u2_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE __attribute((annotate("scalar(range(" PB_XSTR(VAR_v2_MIN) "," PB_XSTR(VAR_v2_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE __attribute((annotate("target('w') scalar(range(" PB_XSTR(VAR_w_MIN) "," PB_XSTR(VAR_w_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE __attribute((annotate("scalar(range(" PB_XSTR(VAR_x_MIN) "," PB_XSTR(VAR_x_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE __attribute((annotate("scalar(range(" PB_XSTR(VAR_y_MIN) "," PB_XSTR(VAR_y_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE __attribute((annotate("scalar(range(" PB_XSTR(VAR_z_MIN) "," PB_XSTR(VAR_z_MAX) ") final)"))), N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(u1),
	      POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2),
	      POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
	      POLYBENCH_ARRAY(z));

  scale_scalar(&alpha, SCALING_FACTOR);
  scale_scalar(&beta, SCALING_FACTOR);
  scale_2d(N, N, POLYBENCH_ARRAY(A), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(u1), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(v1), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(u2), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(v2), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(w), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(x), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(y), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(z), SCALING_FACTOR);

#ifdef COLLECT_STATS
  stats_header();
  stats_scalar("alpha", alpha);
  stats_scalar("beta", beta);
  stats_2d("A", N, N, POLYBENCH_ARRAY(A));
  stats_1d("u1", N, POLYBENCH_ARRAY(u1));
  stats_1d("v1", N, POLYBENCH_ARRAY(v1));
  stats_1d("u2", N, POLYBENCH_ARRAY(u2));
  stats_1d("v2", N, POLYBENCH_ARRAY(v2));
  stats_1d("w", N, POLYBENCH_ARRAY(w));
  stats_1d("x", N, POLYBENCH_ARRAY(x));
  stats_1d("y", N, POLYBENCH_ARRAY(y));
  stats_1d("z", N, POLYBENCH_ARRAY(z));
#endif

#ifndef _LAMP
  /* Start timer. */
  polybench_start_instruments;
#endif

  timer_start();
  /* Run kernel. */
  kernel_gemver (n, alpha, beta,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(u1),
		 POLYBENCH_ARRAY(v1),
		 POLYBENCH_ARRAY(u2),
		 POLYBENCH_ARRAY(v2),
		 POLYBENCH_ARRAY(w),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y),
		 POLYBENCH_ARRAY(z));
  timer_stop();

#ifdef COLLECT_STATS
  stats_scalar("alpha", alpha);
  stats_scalar("beta", beta);
  stats_2d("A", N, N, POLYBENCH_ARRAY(A));
  stats_1d("u1", N, POLYBENCH_ARRAY(u1));
  stats_1d("v1", N, POLYBENCH_ARRAY(v1));
  stats_1d("u2", N, POLYBENCH_ARRAY(u2));
  stats_1d("v2", N, POLYBENCH_ARRAY(v2));
  stats_1d("w", N, POLYBENCH_ARRAY(w));
  stats_1d("x", N, POLYBENCH_ARRAY(x));
  stats_1d("y", N, POLYBENCH_ARRAY(y));
  stats_1d("z", N, POLYBENCH_ARRAY(z));
#endif

#ifndef _LAMP
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(u1);
  POLYBENCH_FREE_ARRAY(v1);
  POLYBENCH_FREE_ARRAY(u2);
  POLYBENCH_FREE_ARRAY(v2);
  POLYBENCH_FREE_ARRAY(w);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(z);
#else
  for (int i = 0; i < n; i++){
    x_float[i] = x[i];
    w_float[i] = w[i];
    for (int j = 0; j < n; j++)
      A_float[i][j] = A[i][j];
  }
#endif

  return 0;
}
