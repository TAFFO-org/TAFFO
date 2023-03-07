/**
 * gesummv.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */
#include "gesummv.cuh"
#include "gesummv_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

extern "C" __global__ void gesummv_kernel(int n, ANN_ALPHA DATA_TYPE alpha, ANN_BETA DATA_TYPE beta, ANN_A DATA_TYPE* A, ANN_B DATA_TYPE* B, ANN_TMP DATA_TYPE* tmp, ANN_X DATA_TYPE* x, ANN_Y DATA_TYPE* y)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < _PB_N)
	{
		int j;
        tmp[i] = 0;
		y[i] = 0;
		for(j = 0; j < _PB_N; j++)
		{	
			tmp[i] += A[i * N + j] * x[j];
			y[i] += B[i * N + j] * x[j];
		}
		y[i] = alpha * tmp[i] + beta  * y[i];
	}
}