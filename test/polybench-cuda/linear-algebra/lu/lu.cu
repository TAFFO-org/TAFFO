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
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((j > k) && (j < _PB_N))
	{
		A[k*N + j] = A[k*N + j] / A[k*N + k];
	}
}


extern "C" __global__ void lu_kernel2(int n, ANN_A DATA_TYPE *A, int k)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i > k) && (j > k) && (i < _PB_N) && (j < _PB_N))
	{
		A[i*N + j] = A[i*N + j] - A[i*N + k] * A[k*N + j];
	}
}
