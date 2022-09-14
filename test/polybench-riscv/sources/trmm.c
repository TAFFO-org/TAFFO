/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trmm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trmm.h"

#ifdef _LAMP
float POLYBENCH_2D(B_float,M,N,m,n);
#endif


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i __attribute__((annotate("scalar(range(0, " PB_XSTR(M) "))")));
  int j __attribute__((annotate("scalar(range(0, " PB_XSTR(N) "))")));

  *alpha = 1.5;
  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {
      if (j < i) {
        A[i][j] = (DATA_TYPE)((i+j) % m)/m;
      } else {
        A[i][j] = 0;
      }
    }
    A[i][i] = 1.0;
    for (j = 0; j < n; j++) {
      B[i][j] = (DATA_TYPE)((n+(i-j)) % n)/n;
    }
 }

}


#ifndef _LAMP
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("B");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, B[i][j]);
    }
  POLYBENCH_DUMP_END("B");
  POLYBENCH_DUMP_FINISH;
}
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_trmm(int m, int n,
		 DATA_TYPE alpha,
		 DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;

//BLAS parameters
//SIDE   = 'L'
//UPLO   = 'L'
//TRANSA = 'T'
//DIAG   = 'U'
// => Form  B := alpha*A**T*B.
// A is MxM
// B is MxN
#pragma scop
  for (i = 0; i < _PB_M; i++)
     for (j = 0; j < _PB_N; j++) {
        for (k = i+1; k < _PB_M; k++)
           B[i][j] += A[k][i] * B[k][j];
        B[i][j] = alpha * B[i][j];
     }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_alpha_MIN) "," PB_XSTR(VAR_alpha_MAX) ") final)"))) alpha;
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_A_MIN) "," PB_XSTR(VAR_A_MAX) ") final)"))),M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE __attribute__((annotate("target('B') scalar(range(" PB_XSTR(VAR_B_MIN) "," PB_XSTR(VAR_B_MAX) ") final)"))),M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  scale_scalar(&alpha, SCALING_FACTOR);
  scale_2d(M,M, POLYBENCH_ARRAY(A), SCALING_FACTOR);
  scale_2d(M,N, POLYBENCH_ARRAY(B), SCALING_FACTOR);

#ifdef COLLECT_STATS
  stats_header();
  stats_scalar("alpha", alpha);
  stats_2d("A", M,M, POLYBENCH_ARRAY(A));
  stats_2d("B", M,N, POLYBENCH_ARRAY(B));
#endif

#ifndef _LAMP
  /* Start timer. */
  polybench_start_instruments;
#endif

  timer_start();
  /* Run kernel. */
  kernel_trmm (m, n, alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
  timer_stop();

#ifdef COLLECT_STATS
  stats_scalar("alpha", alpha);
  stats_2d("A", M,M, POLYBENCH_ARRAY(A));
  stats_2d("B", M,N, POLYBENCH_ARRAY(B));
#endif

#ifndef _LAMP
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(B)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
#else
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      B_float[i][j] = B[i][j];
#endif

  return 0;
}
