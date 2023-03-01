/**
 * bicg.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */
#include "bicg.cuh"
#include "bicg_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>


extern "C" __global__ void bicg_kernel1(int nx, int ny, ANN_A DATA_TYPE *A, ANN_R DATA_TYPE *r, ANN_S DATA_TYPE *s)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < _PB_NY)
	{
		s[j] = 0.0f;

		int i;
		for(i = 0; i < _PB_NX; i++)
		{
			s[j] += r[i] * A[i * NY + j];
		}
	}	
}


extern "C" __global__ void bicg_kernel2(int nx, int ny, ANN_A DATA_TYPE *A, ANN_P DATA_TYPE *p, ANN_Q DATA_TYPE *q)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < _PB_NX)
	{
		q[i] = 0.0f;

		int j;
		for(j=0; j < _PB_NY; j++)
		{
			q[i] += A[i * NY + j] * p[j];
		}
	}
}
