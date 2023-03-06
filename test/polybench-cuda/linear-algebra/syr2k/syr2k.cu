/**
 * syr2k.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include "syr2k.cuh"
#include "syr2k_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

extern "C" __global__ void syr2k_kernel(int ni, int nj, ANN_ALPHA DATA_TYPE alpha, ANN_BETA DATA_TYPE beta, ANN_A DATA_TYPE *a, ANN_B DATA_TYPE *b, ANN_C DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NI))
	{
		c[i * NI + j] *= beta;
		
		int k;
		for(k = 0; k < NJ; k++)
		{
            c[i * NI + j] +=  alpha * a[i * NJ + k] * b[j * NJ + k] + alpha * b[i * NJ + k] * a[j * NJ + k];
		}
	}
}