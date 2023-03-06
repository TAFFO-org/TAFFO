/**
 * mvt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */
#include "mvt.cuh"
#include "mvt_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

extern "C" __global__ void mvt_kernel1(int n, ANN_A DATA_TYPE *a, ANN_X1 DATA_TYPE *x1, ANN_Y_1 DATA_TYPE *y_1)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < _PB_N)
	{
		int j;
		for(j=0; j < _PB_N; j++)
		{
			x1[i] += a[i * N + j] * y_1[j];
		}
	}
}


extern "C" __global__ void mvt_kernel2(int n, ANN_A DATA_TYPE *a, ANN_X2 DATA_TYPE *x2, ANN_Y_2 DATA_TYPE *y_2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < _PB_N)
	{
		int j;
		for(j=0; j < _PB_N; j++)
		{
			x2[i] += a[j * N + i] * y_2[j];	
		}
	}
}