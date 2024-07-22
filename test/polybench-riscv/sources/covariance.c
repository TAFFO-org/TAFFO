/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* covariance.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "covariance.h"


#ifdef _LAMP
DATA_TYPE POLYBENCH_2D(cov_float,M,M,m,m);
#endif

/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
		 DATA_TYPE POLYBENCH_2D(cov,M,M,m,m),
		 DATA_TYPE POLYBENCH_1D(mean,M,m)
               )
{
  int i __attribute__((annotate("scalar(range(0, " PB_XSTR(N) "))")));
  int j __attribute__((annotate("scalar(range(0, " PB_XSTR(M) "))")));
  int k __attribute__((annotate("scalar(range(0, " PB_XSTR(M) "))")));

  *float_n = (DATA_TYPE)n;
  for (j = 0; j < M; j++) {
    mean[j] = 0.0f;
    for (k = 0; k < M; k++){
      cov[j][k] = 0.0f;
    }
  }
  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = ((DATA_TYPE) i*j) / M;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
#if (!defined _LAMP) || (defined _PRINT_OUTPUT)
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(cov,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("cov");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, cov[i][j]);
    }
  POLYBENCH_DUMP_END("cov");
  POLYBENCH_DUMP_FINISH;
}
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_covariance(int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
		       DATA_TYPE POLYBENCH_2D(cov,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i, j, k;

#pragma scop
  for (j = 0; j < _PB_M; j++)
    {
      mean[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        mean[j] += data[i][j];
      mean[j] /= float_n;
    }

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      data[i][j] -= mean[j];

  for (i = 0; i < _PB_M; i++)
    for (j = i; j < _PB_M; j++)
      {
        cov[i][j] = SCALAR_VAL(0.0);
        for (k = 0; k < _PB_N; k++)
	  cov[i][j] += data[k][i] * data[k][j];
        cov[i][j] /= (float_n - SCALAR_VAL(1.0));
        cov[j][i] = cov[i][j];
      }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE __attribute((annotate("target('float_n') scalar(range(" PB_XSTR(VAR_n_MIN) "," PB_XSTR(VAR_n_MAX) ") final)"))) float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE __attribute((annotate("target('data') scalar(range(" PB_XSTR(VAR_data_MIN) "," PB_XSTR(VAR_data_MAX) ") final)"))),N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(cov,DATA_TYPE __attribute((annotate("target('cov') scalar(range(" PB_XSTR(VAR_cov_MIN) "," PB_XSTR(VAR_cov_MAX) ") final)"))),M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE __attribute((annotate("target('mean') scalar(range(" PB_XSTR(VAR_mean_MIN) "," PB_XSTR(VAR_mean_MAX) ") final)"))),M,m);


  /* Initialize array(s). */
  init_array(m, n, &float_n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(cov), POLYBENCH_ARRAY(mean));

#if SCALING_FACTOR!=1
  scale_2d(N, M, POLYBENCH_ARRAY(data), SCALING_FACTOR);
  scale_2d(M, M, POLYBENCH_ARRAY(cov), SCALING_FACTOR);
  scale_1d(M, POLYBENCH_ARRAY(mean), SCALING_FACTOR);
#endif

#ifdef COLLECT_STATS
  stats_header();
  stats_scalar("n", float_n);
  stats_2d("data", N, M, POLYBENCH_ARRAY(data));
  stats_2d("cov", M, M, POLYBENCH_ARRAY(cov));
  stats_1d("mean", M, POLYBENCH_ARRAY(mean));
#endif

#ifndef _LAMP
  /* Start timer. */
  polybench_start_instruments;
#endif

  timer_start();
  /* Run kernel. */
  kernel_covariance(m, n, float_n,
                    POLYBENCH_ARRAY(data),
                    POLYBENCH_ARRAY(cov),
                    POLYBENCH_ARRAY(mean));
  timer_stop();

#ifdef COLLECT_STATS
  stats_scalar("n", float_n);
  stats_2d("data", N, M, POLYBENCH_ARRAY(data));
  stats_2d("cov", M, M, POLYBENCH_ARRAY(cov));
  stats_1d("mean", M, POLYBENCH_ARRAY(mean));
#endif

#ifndef _LAMP
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(cov)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(cov);
  POLYBENCH_FREE_ARRAY(mean);
#else
  for (int i = 0; i < m; i++)
    for (int j = 0; j < m; j++)
      cov_float[i][j] = cov[i][j];
#ifdef _PRINT_OUTPUT
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(cov_float)));
#endif
#endif

  return 0;
}
