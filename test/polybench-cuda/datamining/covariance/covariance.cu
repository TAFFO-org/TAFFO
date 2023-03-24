/**
 * covariance.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */
#include "covariance.cuh"
#include "covariance_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

extern "C" __global__ void mean_kernel(int m, int n, ANN_MEAN DATA_TYPE *mean, ANN_DATA DATA_TYPE *data, ANN_FLOAT_N DATA_TYPE float_n)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < _PB_M)
	{
		mean[j] = 0.0;

		int i;
		for(i = 0; i < _PB_N; i++)
		{
			mean[j] += data[i * M + j];
		}
		mean[j] /= (DATA_TYPE)float_n;
		//mean[j] /= 3214212.01;
	}
}


extern "C" __global__ void reduce_kernel(int m, int n, ANN_MEAN DATA_TYPE *mean, ANN_DATA DATA_TYPE *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
		
	if ((i < _PB_N) && (j < _PB_M))
	{
		data[i * M + j] -= mean[j];	
	}
}


extern "C" __global__ void covar_kernel(int m, int n, ANN_SYMMAT DATA_TYPE *symmat, ANN_DATA DATA_TYPE *data)
{
	int j1 = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j2;

	if (j1 < _PB_M)
	{
		for (j2 = j1; j2 < _PB_M; j2++)
		{		
			symmat[j1*M + j2] = 0.0;
			for(i = 0; i < _PB_N; i++)
			{
				symmat[j1 * M + j2] += data[i * M + j1] * data[i * M + j2];
			}
			symmat[j2 * M + j1] = symmat[j1 * M + j2];
		}
	}
}