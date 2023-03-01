/**
 * 2mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */
#include "2mm.cuh"
#include "2mm_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>


extern "C" __global__ void mm2_kernel1(int ni, int nj, int nk, int nl, ANN_ALPHA DATA_TYPE alpha, ANN_BETA DATA_TYPE beta, ANN_TMP DATA_TYPE *tmp, ANN_A DATA_TYPE *A, ANN_B DATA_TYPE *B)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_NI) && (j < _PB_NJ))
	{ 
		tmp[i * NJ + j] = 0;
		int k;
		for (k = 0; k < _PB_NK; k++)
		{
			tmp[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
		}
	}
}


extern "C" __global__ void mm2_kernel2(int ni, int nj, int nk, int nl, ANN_ALPHA DATA_TYPE alpha, ANN_BETA DATA_TYPE beta, ANN_TMP DATA_TYPE *tmp, ANN_C DATA_TYPE *C, ANN_D DATA_TYPE *D)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_NI) && (j < _PB_NL))
	{ 
		D[i * NL + j] *= beta;
		int k;
		for (k = 0; k < _PB_NJ; k++)
		{
			D[i * NL + j] += tmp[i * NJ + k] * C[k * NL + j];
		}
	}
}
