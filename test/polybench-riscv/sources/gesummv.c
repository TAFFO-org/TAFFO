/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gesummv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gesummv.h"

#ifdef _LAMP
/* Retrieve problem size. */
int n = N;

/* Variable declaration/allocation. */
DATA_TYPE alpha;
DATA_TYPE beta;
DATA_TYPE POLYBENCH_2D(A, N, N, n, n);
DATA_TYPE POLYBENCH_2D(B, N, N, n, n);
DATA_TYPE POLYBENCH_1D(tmp, N, n);
DATA_TYPE POLYBENCH_1D(x, N, n);
DATA_TYPE POLYBENCH_1D(y, N, n);

float POLYBENCH_1D(y_float, N, n);
#endif


/* Array initialization. */
static
void init_array()
{
  int i __attribute__((annotate("scalar(range(0," PB_XSTR(N) "))")));
  int j __attribute__((annotate("scalar(range(0," PB_XSTR(N) "))")));

  alpha = 1.5;
  beta = 1.2;
  for (i = 0; i < n; i++)
    {
      x[i] = (DATA_TYPE)( i % n) / n;
      for (j = 0; j < n; j++) {
	A[i][j] = (DATA_TYPE) ((i*j+1) % n) / n;
	B[i][j] = (DATA_TYPE) ((i*j+2) % n) / n;
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
void kernel_gesummv()
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
      tmp[i] = SCALAR_VAL(0.0);
      y[i] = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_N; j++)
	{
	  tmp[i] = A[i][j] * x[j] + tmp[i];
	  y[i] = B[i][j] * x[j] + y[i];
	}
      y[i] = alpha * tmp[i] + beta * y[i];
    }
#pragma endscop

}


int main(int argc, char** argv)
{
#ifndef _LAMP
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE __attribute__((annotate("scalar()"))) alpha;
  DATA_TYPE __attribute__((annotate("scalar()"))) beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE __attribute__((annotate("scalar()"))), N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE __attribute__((annotate("scalar()"))), N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE __attribute__((annotate("scalar(range(-256, 255) final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE __attribute__((annotate("scalar()"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE __attribute__((annotate("target('y') scalar(range(-256, 255) final)"))), N, n);
#endif


  /* Initialize array(s). */
  init_array ();

#ifndef _LAMP
  /* Start timer. */
  polybench_start_instruments;
#endif

  /* Run kernel. */
  kernel_gesummv ();

#ifndef _LAMP
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
#else
  for (int i = 0; i < n; i++)
      y_float[i] = y[i];
#endif

  return 0;
}
