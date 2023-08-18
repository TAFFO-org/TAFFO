/**
 * 3mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */
#include "3mm.cuh"
#include "3mm_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>


extern "C" __global__ void mm3_kernel1(int ni, int nj, int nk, int nl, int nm, ANN_A DATA_TYPE *A, ANN_B DATA_TYPE *B, ANN_E DATA_TYPE *E)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_NI) && (j < _PB_NJ))
	{
		E[i * NJ + j] = 0;
		int k;
		for(k=0; k < _PB_NK; k++)
		{
			E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
		}
	}
}

	
extern "C" __global__ void mm3_kernel2(int ni, int nj, int nk, int nl, int nm,  ANN_C DATA_TYPE *C, ANN_D DATA_TYPE *D, ANN_F DATA_TYPE *F)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_NJ) && (j < _PB_NL))
	{
		F[i * NL + j] = 0;
		int k;
		for(k=0; k < _PB_NM; k++)
		{
			F[i * NL + j] += C[i * NM + k] * D[k * NL +j];
		}
	}
}

	
extern "C" __global__ void mm3_kernel3(int ni, int nj, int nk, int nl, int nm, ANN_E DATA_TYPE *E, ANN_F DATA_TYPE *F, ANN_G DATA_TYPE *G)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_NI) && (j < _PB_NL))
	{
		G[i * NL + j] = 0;
		int k;
		for(k=0; k < _PB_NJ; k++)
		{
			ANN_G DATA_TYPE tmp = E[i * NJ + k] * F[k * NL + j];
			G[i * NL + j] += tmp;
		}
	}
}