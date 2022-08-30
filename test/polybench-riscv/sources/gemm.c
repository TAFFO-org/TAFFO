/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemm.h"

#ifdef _LAMP
/* Retrieve problem size. */
int ni = NI;
int nj = NJ;
int nk = NK;

/* Variable declaration/allocation. */
DATA_TYPE alpha;
DATA_TYPE beta;
DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj);
DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk);
DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj);

float POLYBENCH_2D(C_float,NI,NJ,ni,nj);
#endif


/* Array initialization. */
static
void init_array()
{
  int i, j;

  alpha = 1.5;
  beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+2) % nj) / nj;
}


#ifndef _LAMP
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}
#endif


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm()
{
  int i, j, k;

//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
#pragma scop
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NJ; j++)
	C[i][j] *= beta;
    for (k = 0; k < _PB_NK; k++) {
       for (j = 0; j < _PB_NJ; j++)
	  C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
#ifndef _LAMP
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE __attribute((annotate("scalar()"))) alpha;
  DATA_TYPE __attribute((annotate("scalar()"))) beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE __attribute((annotate("target('C') scalar()"))),NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE __attribute((annotate("scalar(range(-64, 64))"))),NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE __attribute((annotate("scalar()"))),NK,NJ,nk,nj);
#endif


  /* Initialize array(s). */
  init_array ();

#ifndef _LAMP
  /* Start timer. */
  polybench_start_instruments;
#endif

  /* Run kernel. */
  kernel_gemm ();

#ifndef _LAMP
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
#else
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++)
      C_float[i][j] = C[i][j];
#endif

  return 0;
}
