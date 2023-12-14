/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* syrk.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syrk.h"


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i __attribute__((annotate("scalar(range(0, " PB_XSTR(N) "))")));
  int j __attribute__((annotate("scalar(range(0, " PB_XSTR(M) "))")));

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_syrk(int n, int m,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j, k;

//BLAS PARAMS
//TRANS = 'N'
//UPLO  = 'L'
// =>  Form  C := alpha*A*A**T + beta*C.
//A is NxM
//C is NxN
#pragma scop
  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < _PB_M; k++) {
      for (j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
    }
  }
#pragma endscop

}


int BENCH_MAIN(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_alpha_MIN) "," PB_XSTR(VAR_alpha_MAX) "))"))) alpha;
  DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_beta_MIN) "," PB_XSTR(VAR_beta_MAX) "))"))) beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE __attribute__((annotate("target('C') scalar(range(" PB_XSTR(VAR_C_MIN) "," PB_XSTR(VAR_C_MAX) "))"))),N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_A_MIN) "," PB_XSTR(VAR_A_MAX) "))"))),N,M,n,m);

  for (int benchmark_i = 0; benchmark_i < BENCH_NUM_ITERATIONS; benchmark_i++) {
    /* Initialize array(s). */
    init_array(n, m, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

    srand(POLYBENCH_RANDOM_SEED);
    randomize_scalar(&alpha, POLYBENCH_RANDOMIZE_RANGE);
    randomize_scalar(&beta, POLYBENCH_RANDOMIZE_RANGE);
    randomize_2d(N, N, C, POLYBENCH_RANDOMIZE_RANGE);
    randomize_2d(N, M, A, POLYBENCH_RANDOMIZE_RANGE);

#if SCALING_FACTOR!=1
  scale_scalar(&alpha, SCALING_FACTOR);
  scale_scalar(&beta, SCALING_FACTOR);
  scale_2d(N,N, POLYBENCH_ARRAY(C), SCALING_FACTOR);
  scale_2d(N,M, POLYBENCH_ARRAY(A), SCALING_FACTOR);
#endif

#ifdef COLLECT_STATS
  stats_header();
  stats_scalar("alpha", alpha);
  stats_scalar("beta", beta);
  stats_2d("C", N,N, POLYBENCH_ARRAY(C));
  stats_2d("A", N,M, POLYBENCH_ARRAY(A));
#endif

    /* Start timer. */
    polybench_start_instruments;

    /* Run kernel. */
    kernel_syrk(n, m, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;
  }

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}
