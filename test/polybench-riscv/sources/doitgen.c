/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* doitgen.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "doitgen.h"

#ifdef _LAMP
float POLYBENCH_3D(A_float,NR,NQ,NP,nr,nq,np);
#endif


/* Array initialization. */
static
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
                DATA_TYPE POLYBENCH_1D(sum,NP,np))
{
  int i __attribute__((annotate("scalar(range(0, " PB_XSTR(NP) "))")));
  int j __attribute__((annotate("scalar(range(0, " PB_XSTR(NP) "))")));
  int k __attribute__((annotate("scalar(range(0, " PB_XSTR(NP) "))")));

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
	A[i][j][k] = (DATA_TYPE) ((i*j + k)%np) / np;
  for (i = 0; i < np; i++) {
    sum[i] = 0;
    for (j = 0; j < np; j++) {
      C4[i][j] = (DATA_TYPE)(i * j % np) / np;
    }
  }
}


#ifndef _LAMP
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
	if ((i*nq*np+j*np+k) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_doitgen(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_1D(sum,NP,np))
{
  int r, q, p, s;

#pragma scop
  for (r = 0; r < _PB_NR; r++)
    for (q = 0; q < _PB_NQ; q++)  {
      for (p = 0; p < _PB_NP; p++)  {
	sum[p] = SCALAR_VAL(0.0);
	for (s = 0; s < _PB_NP; s++)
	  sum[p] += A[r][q][s] * C4[s][p];
      }
      for (p = 0; p < _PB_NP; p++)
	A[r][q][p] = sum[p];
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int nr = NR;
  int nq = NQ;
  int np = NP;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE __attribute__((annotate("target('A') scalar(range(" PB_XSTR(VAR_A_MIN) "," PB_XSTR(VAR_A_MAX) ") final)"))),NR,NQ,NP,nr,nq,np);
  POLYBENCH_1D_ARRAY_DECL(sum,DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_sum_MIN) "," PB_XSTR(VAR_sum_MAX) ") final)"))),NP,np);
  POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_C4_MIN) "," PB_XSTR(VAR_C4_MAX) ") final)"))),NP,NP,np,np);

  /* Initialize array(s). */
  init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4),
	      POLYBENCH_ARRAY(sum)
             );

  scale_3d(NR,NQ,NP, POLYBENCH_ARRAY(A), SCALING_FACTOR);
  scale_1d(NP, POLYBENCH_ARRAY(sum), SCALING_FACTOR);
  scale_2d(NP,NP, POLYBENCH_ARRAY(C4), SCALING_FACTOR);

#ifdef COLLECT_STATS
  stats_header();
  stats_3d("A", NR, NQ, NP, POLYBENCH_ARRAY(A));
  stats_1d("sum", NP, POLYBENCH_ARRAY(sum));
  stats_2d("C4", NP, NP, POLYBENCH_ARRAY(C4));
#endif

#ifndef _LAMP
  /* Start timer. */
  polybench_start_instruments;
#endif

  timer_start();
  /* Run kernel. */
  kernel_doitgen (nr, nq, np,
		  POLYBENCH_ARRAY(A),
		  POLYBENCH_ARRAY(C4),
		  POLYBENCH_ARRAY(sum));
  timer_stop();

#ifdef COLLECT_STATS
  stats_3d("A", NR, NQ, NP, POLYBENCH_ARRAY(A));
  stats_1d("sum", NP, POLYBENCH_ARRAY(sum));
  stats_2d("C4", NP, NP, POLYBENCH_ARRAY(C4));
#endif

#ifndef _LAMP
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nr, nq, np,  POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(sum);
  POLYBENCH_FREE_ARRAY(C4);
#else
  for (int i = 0; i < nr; i++)
    for (int j = 0; j < nq; j++)
      for (int k = 0; k < np; k++)
	      A_float[i][j][k] = A[i][j][k];
#endif

  return 0;
}
