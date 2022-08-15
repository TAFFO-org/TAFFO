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
/* Retrieve problem size. */
int n = N;

/* Variable declaration/allocation. */
DATA_TYPE alpha;
DATA_TYPE beta;
DATA_TYPE POLYBENCH_2D(A, N, N, n, n);
DATA_TYPE POLYBENCH_1D(u1, N, n);
DATA_TYPE POLYBENCH_1D(v1, N, n);
DATA_TYPE POLYBENCH_1D(u2, N, n);
DATA_TYPE POLYBENCH_1D(v2, N, n);
DATA_TYPE POLYBENCH_1D(w, N, n);
DATA_TYPE POLYBENCH_1D(x, N, n);
DATA_TYPE POLYBENCH_1D(y, N, n);
DATA_TYPE POLYBENCH_1D(z, N, n);

float POLYBENCH_2D(A_float, N, N, n, n);
float POLYBENCH_1D(w_float, N, n);
float POLYBENCH_1D(x_float, N, n);
#endif


/* Array initialization. */
static
void init_array ()
{
  int i, j;

  alpha = 1.5;
  beta = 1.2;

  DATA_TYPE fn = (DATA_TYPE)n;

  DATA_TYPE constTwoVal = 2.0f;
  DATA_TYPE constFourVal = 4.0f;
  DATA_TYPE constSixVal = 6.0f;
  DATA_TYPE constEightVal = 8.0f;
  DATA_TYPE constNineVal = 9.0f;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/constTwoVal;
      v1[i] = ((i+1)/fn)/constFourVal;
      v2[i] = ((i+1)/fn)/constSixVal;
      y[i] = ((i+1)/fn)/constEightVal;
      z[i] = ((i+1)/fn)/constNineVal;
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
void kernel_gemver()
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
#ifndef _LAMP
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);
#endif

  /* Initialize array(s). */
  init_array ();

#ifndef _LAMP
  /* Start timer. */
  polybench_start_instruments;
#endif

  /* Run kernel. */
  kernel_gemver ();

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
