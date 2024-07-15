/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* correlation.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "correlation.h"

#ifdef _LAMP
DATA_TYPE POLYBENCH_2D(corr_float,M,M,m,m);
#endif


/* Array initialization. */
static
void init_array (int m,
		 int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
		 DATA_TYPE POLYBENCH_2D(corr,M,M,n,m),
		 DATA_TYPE POLYBENCH_1D(mean,M,m),
		 DATA_TYPE POLYBENCH_1D(stddev,M,m)
               )
{
  int __attribute((annotate("scalar(range(0, 260) final disabled)"))) i;
  int __attribute((annotate("scalar(range(0, 240) final disabled)"))) j;
  int __attribute((annotate("scalar(range(0, 240) final disabled)"))) k;
//, POLYBENCH_ARRAY(corr), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev)
  *float_n = (DATA_TYPE)N;

  for (j = 0; j < M; j++) {
    mean[j] = 0.0f;
    stddev[j] = 0.0f;
    for (k = 0; k < M; k++){
      corr[j][k] = 0.0f;
    }
  }
  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = ((DATA_TYPE)(i*j)/M + i)/N;

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
#if (!defined _LAMP) || (defined _PRINT_OUTPUT)
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(corr,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("corr");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, corr[i][j]);
    }
  POLYBENCH_DUMP_END("corr");
  POLYBENCH_DUMP_FINISH;
}
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_correlation(int m, int n,
			DATA_TYPE float_n,
			DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
			DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
			DATA_TYPE POLYBENCH_1D(mean,M,m),
			DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
  int i, j, k;

  DATA_TYPE __attribute((annotate("scalar()"))) eps = SCALAR_VAL(0.1);


#pragma scop
  for (j = 0; j < _PB_M; j++)
    {
      mean[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }


   for (j = 0; j < _PB_M; j++)
    {
      stddev[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = SQRT_FUN(stddev[j]);
      /* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
      stddev[j] = stddev[j] <= eps ? SCALAR_VAL(1.0) : stddev[j];
    }

  /* Center and reduce the column vectors. */
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= SQRT_FUN(float_n) * stddev[j];
      }

  /* Calculate the m * m correlation matrix. */
  for (i = 0; i < _PB_M-1; i++)
    {
      corr[i][i] = SCALAR_VAL(1.0);
      for (j = i+1; j < _PB_M; j++)
        {
          corr[i][j] = SCALAR_VAL(0.0);
          for (k = 0; k < _PB_N; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[_PB_M-1][_PB_M-1] = SCALAR_VAL(1.0);
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE __attribute((annotate("scalar(range(" PB_XSTR(VAR_n_MIN) "," PB_XSTR(VAR_n_MAX) "))"))) float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE __attribute((annotate("scalar(range(" PB_XSTR(VAR_data_MIN) "," PB_XSTR(VAR_data_MAX) ") final)"))),N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(corr,DATA_TYPE __attribute((annotate("target('corr') scalar()"))),M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE __attribute((annotate("scalar()"))),M,m);
  POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE __attribute((annotate("scalar(range(" PB_XSTR(VAR_stddev_MIN) "," PB_XSTR(VAR_stddev_MAX) ") final)"))),M,m);

  /* Initialize array(s). */
  init_array(m, n, &float_n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(corr), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev));

#if SCALING_FACTOR!=1
  scale_2d(N, M, POLYBENCH_ARRAY(data), SCALING_FACTOR);
  scale_2d(M, M, POLYBENCH_ARRAY(corr), SCALING_FACTOR);
  scale_1d(M, POLYBENCH_ARRAY(mean), SCALING_FACTOR);
  scale_1d(M, POLYBENCH_ARRAY(stddev), SCALING_FACTOR);
#endif

#ifdef COLLECT_STATS
  stats_header();
  stats_scalar("n", float_n);
  stats_2d("data", N, M, POLYBENCH_ARRAY(data));
  stats_2d("corr", M, M, POLYBENCH_ARRAY(corr));
  stats_1d("mean", M, POLYBENCH_ARRAY(mean));
  stats_1d("stddev", M, POLYBENCH_ARRAY(stddev));
#endif

#ifndef _LAMP
  /* Start timer. */
  polybench_start_instruments;
#endif

  timer_start();

  /* Run kernel. */
  kernel_correlation(m, n, float_n,
                     POLYBENCH_ARRAY(data),
                     POLYBENCH_ARRAY(corr),
                     POLYBENCH_ARRAY(mean),
                     POLYBENCH_ARRAY(stddev));
  timer_stop();

#ifdef COLLECT_STATS
  stats_scalar("n", float_n);
  stats_2d("data", N, M, POLYBENCH_ARRAY(data));
  stats_2d("corr", M, M, POLYBENCH_ARRAY(corr));
  stats_1d("mean", M, POLYBENCH_ARRAY(mean));
  stats_1d("stddev", M, POLYBENCH_ARRAY(stddev));
#endif

#ifndef _LAMP
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(corr)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(corr);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);
#else
  for (int i = 0; i < m; i++)
    for (int j = 0; j < m; j++)
      corr_float[i][j] = corr[i][j];
#ifdef _PRINT_OUTPUT
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(corr_float)));
#endif
#endif

  return 0;
}
