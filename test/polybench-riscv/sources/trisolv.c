/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trisolv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trisolv.h"

#ifdef _LAMP
float POLYBENCH_1D(x_float, N, n);
#endif

/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i __attribute__((annotate("scalar(range(0," PB_XSTR(N) ") final)")));
  int j __attribute__((annotate("scalar(range(0," PB_XSTR(N) ") final)")));

  for (i = 0; i < n; i++)
    {
      x[i] = 0; //- 999;
      b[i] =  i ;
      for (j = 0; j < n; j++)
        if (j <= i) {
          L[i][j] = (DATA_TYPE) (i+n-j+1)*2/n;
        } else {
          L[i][j] = 0;
        }
    }
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
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_trisolv(int n,
		    DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
      x[i] = b[i];
      for (j = 0; j <i; j++)
        x[i] -= L[i][j] * x[j];
      x[i] = x[i] / L[i][i];
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(L, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_L_MIN) "," PB_XSTR(VAR_L_MAX) ") final)"))), N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE __attribute__((annotate("target('x') scalar(range(" PB_XSTR(VAR_x_MIN) "," PB_XSTR(VAR_x_MAX) ") final)"))), N, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_b_MIN) "," PB_XSTR(VAR_b_MAX) ") final)"))), N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  scale_2d(N, N, POLYBENCH_ARRAY(L), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(x), SCALING_FACTOR);
  scale_1d(N, POLYBENCH_ARRAY(b), SCALING_FACTOR);

#ifdef COLLECT_STATS
  stats_header();
  stats_2d("L", N, N, POLYBENCH_ARRAY(L));
  stats_1d("x", N, POLYBENCH_ARRAY(x));
  stats_1d("b", N, POLYBENCH_ARRAY(b));
#endif

#ifndef _LAMP
  /* Start timer. */
  polybench_start_instruments;
#endif

  timer_start();
  /* Run kernel. */
  kernel_trisolv (n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));
  timer_stop();

#ifdef COLLECT_STATS
  stats_2d("L", N, N, POLYBENCH_ARRAY(L));
  stats_1d("x", N, POLYBENCH_ARRAY(x));
  stats_1d("b", N, POLYBENCH_ARRAY(b));
#endif

#ifndef _LAMP
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(L);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(b);
#else
  for (int i=0; i<N; i++)
    x_float[i] = x[i];
#endif

  return 0;
}
