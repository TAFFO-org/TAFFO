/**
 * gemver.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */
#include "gemver.cuh"
#include "gemver_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

extern "C"  __global__ void gemver_kernel1(int n, ANN_ALPHA DATA_TYPE alpha, ANN_BETA DATA_TYPE beta, ANN_A DATA_TYPE *a, ANN_V1 DATA_TYPE *v1, ANN_V2 DATA_TYPE *v2, ANN_U1 DATA_TYPE *u1, ANN_U2 DATA_TYPE *u2)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_N) && (j < _PB_N))
	{
		a[i * N + j] += u1[i] * v1[j] + u2[i] * v2[j];
	}
}


extern "C"  __global__ void gemver_kernel2(int n, ANN_ALPHA DATA_TYPE alpha, ANN_BETA DATA_TYPE beta, ANN_A DATA_TYPE *a, ANN_X DATA_TYPE *x, ANN_Y DATA_TYPE *y, ANN_Z DATA_TYPE *z)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < _PB_N)
	{
		int j;
		for(j = 0; j < _PB_N; j++) 
		{
			x[i] += beta * a[j * N + i] * y[j];
		}
		x[i] += z[i];
	}
}


extern "C"  __global__ void gemver_kernel3(int n, ANN_ALPHA DATA_TYPE alpha, ANN_BETA DATA_TYPE beta, ANN_A DATA_TYPE *a, ANN_X DATA_TYPE *x, ANN_W DATA_TYPE *w)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i >= 0) && (i < _PB_N))
	{
		int j;
		for(j = 0; j < _PB_N; j++)
		{ 
			ANN_W DATA_TYPE tmp = alpha * a[i*N + j] * x[j];
            w[i] += tmp;
		}
	}
}