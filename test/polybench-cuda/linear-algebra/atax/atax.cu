/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */
#include "atax.cuh"
#include "atax_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

extern "C" __global__ void atax_kernel1(int nx, int ny, ANN_A DATA_TYPE *A, ANN_X DATA_TYPE *x, ANN_TMP DATA_TYPE *tmp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < _PB_NX)
	{
		tmp[i] = 0;
		int j;
		for(j=0; j < _PB_NY; j++)
		{
			tmp[i] += A[i*NY+j] * x[j];
		}
	}
}

extern "C" __global__ void atax_kernel2(int nx, int ny, ANN_A DATA_TYPE *A, ANN_Y DATA_TYPE *y, ANN_TMP DATA_TYPE *tmp)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < _PB_NY)
	{
		y[j] = 0;
		int i;
		for(i=0; i < _PB_NX; i++)
		{
			y[j] += A[i*NY+j] * tmp[i];
		}
	}
}