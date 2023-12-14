/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* 2mm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "2mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl),
                DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj)
               )
{
  int i __attribute__((annotate("scalar(range(0, " PB_XSTR(NK) "))")));
  int j __attribute__((annotate("scalar(range(0, " PB_XSTR(NL) "))")));

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (DATA_TYPE) ((i*(j+3)+1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) (i*(j+2) % nk) / nk;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      tmp[i][j] = 0;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D[i][j]);
    }
  POLYBENCH_DUMP_END("D");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_2mm(int ni, int nj, int nk, int nl,
		DATA_TYPE alpha,
		DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j, k;

#pragma scop
  /* D := alpha*A*B*C + beta*D */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NJ; j++)
      {
	tmp[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < _PB_NK; ++k)
	  tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NL; j++)
      {
	D[i][j] *= beta;
	for (k = 0; k < _PB_NJ; ++k)
	  D[i][j] += tmp[i][k] * C[k][j];
      }
#pragma endscop

}


int BENCH_MAIN(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  /* Variable declaration/allocation. */
  DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_alpha_MIN) "," PB_XSTR(VAR_alpha_MAX) "))"))) alpha;
  DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_beta_MIN) "," PB_XSTR(VAR_beta_MAX) "))"))) beta;
  POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE __attribute__((annotate("scalar()"))),NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_A_MIN) "," PB_XSTR(VAR_A_MAX) "))"))),NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_B_MIN) "," PB_XSTR(VAR_B_MAX) "))"))),NK,NJ,nk,nj);
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_C_MIN) "," PB_XSTR(VAR_C_MAX) "))"))),NJ,NL,nj,nl);
  POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE __attribute__((annotate("target('D') scalar(range(" PB_XSTR(VAR_D_MIN) "," PB_XSTR(VAR_D_MAX) "))"))),NI,NL,ni,nl);

  for (int benchmark_i = 0; benchmark_i < BENCH_NUM_ITERATIONS; benchmark_i++) {
  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D),
              POLYBENCH_ARRAY(tmp));

    srand(POLYBENCH_RANDOM_SEED);
    randomize_scalar(&alpha, POLYBENCH_RANDOMIZE_RANGE);
    randomize_scalar(&beta, POLYBENCH_RANDOMIZE_RANGE);
    randomize_2d(NI, NK, A, POLYBENCH_RANDOMIZE_RANGE);
    randomize_2d(NK, NJ, B, POLYBENCH_RANDOMIZE_RANGE);
    randomize_2d(NJ, NL, C, POLYBENCH_RANDOMIZE_RANGE);
    randomize_2d(NI, NL, D, POLYBENCH_RANDOMIZE_RANGE);

#if SCALING_FACTOR!=1
    scale_scalar(&alpha, SCALING_FACTOR);
    scale_scalar(&beta, SCALING_FACTOR);
    scale_2d(NI, NK, POLYBENCH_ARRAY(A), SCALING_FACTOR);
    scale_2d(NK, NJ, POLYBENCH_ARRAY(B), SCALING_FACTOR);
    scale_2d(NJ, NL, POLYBENCH_ARRAY(C), SCALING_FACTOR);
    scale_2d(NI, NL, POLYBENCH_ARRAY(D), SCALING_FACTOR);
    scale_2d(NI, NJ, POLYBENCH_ARRAY(tmp), SCALING_FACTOR);
#endif

#ifdef COLLECT_STATS
    stats_header();
    stats_scalar("alpha", alpha);
    stats_scalar("beta", beta);
    stats_2d("A", NI, NK, POLYBENCH_ARRAY(A));
    stats_2d("B", NK, NJ, POLYBENCH_ARRAY(B));
    stats_2d("C", NJ, NL, POLYBENCH_ARRAY(C));
    stats_2d("D", NI, NL, POLYBENCH_ARRAY(D));
    stats_2d("tmp", NI, NJ, POLYBENCH_ARRAY(tmp));
#endif

    /* Start timer. */
    polybench_start_instruments;

    /* Run kernel. */
    kernel_2mm(ni, nj, nk, nl,
               alpha, beta,
               POLYBENCH_ARRAY(tmp),
               POLYBENCH_ARRAY(A),
               POLYBENCH_ARRAY(B),
               POLYBENCH_ARRAY(C),
               POLYBENCH_ARRAY(D));

    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;
  }

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(D)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);

  return 0;
}
