/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* 3mm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "3mm.h"

#ifdef _LAMP
float POLYBENCH_2D (G_float, NI, NL, ni, nl);
#endif


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl))
{
  int i __attribute__((annotate("scalar(range(0, " PB_XSTR(NM) "))")));
  int j __attribute__((annotate("scalar(range(0, " PB_XSTR(NM) "))")));

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / (5*ni);
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) ((i*(j+1)+2) % nj) / (5*nj);
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = (DATA_TYPE) (i*(j+3) % nl) / (5*nl);
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) ((i*(j+2)+2) % nk) / (5*nk);
}


#ifndef _LAMP
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("G");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, G[i][j]);
    }
  POLYBENCH_DUMP_END("G");
  POLYBENCH_DUMP_FINISH;
}
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j, k;

#pragma scop
  /* E := A*B */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NJ; j++)
      {
	E[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < _PB_NK; ++k)
	  E[i][j] += A[i][k] * B[k][j];
      }
  /* F := C*D */
  for (i = 0; i < _PB_NJ; i++)
    for (j = 0; j < _PB_NL; j++)
      {
	F[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < _PB_NM; ++k)
	  F[i][j] += C[i][k] * D[k][j];
      }
  /* G := E*F */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NL; j++)
      {
	G[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < _PB_NJ; ++k)
	  G[i][j] += E[i][k] * F[k][j];
      }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_E_MIN) "," PB_XSTR(VAR_E_MAX) ") final)"))), NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_A_MIN) "," PB_XSTR(VAR_A_MAX) ") final)"))), NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_B_MIN) "," PB_XSTR(VAR_B_MAX) ") final)"))), NK, NJ, nk, nj);
  POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_F_MIN) "," PB_XSTR(VAR_F_MAX) ") final)"))), NJ, NL, nj, nl);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_C_MIN) "," PB_XSTR(VAR_C_MAX) ") final)"))), NJ, NM, nj, nm);
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_D_MIN) "," PB_XSTR(VAR_D_MAX) ") final)"))), NM, NL, nm, nl);
  POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE __attribute__((annotate("target('G') scalar(range(" PB_XSTR(VAR_G_MIN) "," PB_XSTR(VAR_G_MAX) ") final)"))), NI, NL, ni, nl);


  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  scale_2d(NI, NJ, POLYBENCH_ARRAY(E), SCALING_FACTOR);
  scale_2d(NI, NK, POLYBENCH_ARRAY(A), SCALING_FACTOR);
  scale_2d(NK, NJ, POLYBENCH_ARRAY(B), SCALING_FACTOR);
  scale_2d(NJ, NL, POLYBENCH_ARRAY(F), SCALING_FACTOR);
  scale_2d(NJ, NM, POLYBENCH_ARRAY(C), SCALING_FACTOR);
  scale_2d(NM, NL, POLYBENCH_ARRAY(D), SCALING_FACTOR);
  scale_2d(NI, NL, POLYBENCH_ARRAY(G), SCALING_FACTOR);

#ifdef COLLECT_STATS
  stats_header();
  stats_2d("E", NI, NJ, POLYBENCH_ARRAY(A));
  stats_2d("A", NI, NK, POLYBENCH_ARRAY(B));
  stats_2d("B", NK, NJ, POLYBENCH_ARRAY(C));
  stats_2d("F", NJ, NL, POLYBENCH_ARRAY(D));
  stats_2d("C", NJ, NM, POLYBENCH_ARRAY(E));
  stats_2d("D", NM, NL, POLYBENCH_ARRAY(F));
  stats_2d("G", NI, NL, POLYBENCH_ARRAY(G));
#endif

#ifndef _LAMP
  /* Start timer. */
  polybench_start_instruments;
#endif

  timer_start();
  /* Run kernel. */
  kernel_3mm (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(E),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(F),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D),
	      POLYBENCH_ARRAY(G));
  timer_stop();

#ifdef COLLECT_STATS
  stats_2d("E", NI, NJ, POLYBENCH_ARRAY(A));
  stats_2d("A", NI, NK, POLYBENCH_ARRAY(B));
  stats_2d("B", NK, NJ, POLYBENCH_ARRAY(C));
  stats_2d("F", NJ, NL, POLYBENCH_ARRAY(D));
  stats_2d("C", NJ, NM, POLYBENCH_ARRAY(E));
  stats_2d("D", NM, NL, POLYBENCH_ARRAY(F));
  stats_2d("G", NI, NL, POLYBENCH_ARRAY(G));
#endif

#ifndef _LAMP
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(G)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(E);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(F);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(G);
#else
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nl; j++)
      G_float[i][j] = G[i][j];
#endif

  return 0;
}
