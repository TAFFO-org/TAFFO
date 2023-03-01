/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */
#include "gemm.cuh"
#include "gemm_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>


extern "C" __global__ void gemm_kernel(int ni, int nj, int nk, ANN_ALPHA DATA_TYPE alpha, ANN_BETA DATA_TYPE beta, ANN_A DATA_TYPE *a, ANN_B DATA_TYPE *b, ANN_C DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_NI) && (j < _PB_NJ))
	{	
		ANN_C DATA_TYPE tmp = beta * c[i * NJ + j];
        c[i * NJ + j] = tmp;
		int k;
		for(k=0; k < _PB_NK; k++)
		{
			c[i * NJ + j] += alpha * a[i * NK + k] * b[k * NJ +j];
		}
	}
}