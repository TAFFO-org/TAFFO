/**
 * jacobi1D.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */
#include "jacobi1D.cuh"
#include "jacobi1D_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

extern "C" __global__ void runJacobiCUDA_kernel1(int n, ANN_A DATA_TYPE* A, ANN_B DATA_TYPE* B)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i > 1) && (i < (_PB_N-1)))
	{
		B[i] = 0.33333f * (A[i-1] + A[i] + A[i + 1]);
	}
}


extern "C" __global__ void runJacobiCUDA_kernel2(int n, ANN_A DATA_TYPE* A, ANN_B DATA_TYPE* B)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((j > 1) && (j < (_PB_N-1)))
	{
		A[j] = B[j];
	}
}