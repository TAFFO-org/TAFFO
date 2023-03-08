/**
 * lu.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */
#include "lu.cuh"
#include "lu_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>


extern "C" __global__ void lu_kernel1(int n, ANN_A DATA_TYPE *A, int k)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + (k + 1);
	
	if ((j < _PB_N))
	{
		A[k*N + j] = A[k*N + j] / A[k*N + k];
	}
}


extern "C" __global__ void lu_kernel2(int n, ANN_A DATA_TYPE *A, int k)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + (k + 1);
	int i = blockIdx.y * blockDim.y + threadIdx.y + (k + 1);
	
	if ((i < n) && (j < n))
	{
		__attribute__((annotate("scalar()"))) DATA_TYPE tmp = A[i*n + k] * A[k*n + j];
		A[i*n + j] = A[i*n + j] - tmp;
	}
}
