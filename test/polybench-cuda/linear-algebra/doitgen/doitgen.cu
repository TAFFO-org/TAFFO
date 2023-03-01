/**
 * doitgen.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */
#include "doitgen.cuh"
#include "doitgen_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

extern "C" __global__ void doitgen_kernel1(int nr, int nq, int np, ANN_SUM DATA_TYPE *sum, ANN_A DATA_TYPE *A, ANN_C4 DATA_TYPE *C4, int r)
{
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	int q = blockIdx.y * blockDim.y + threadIdx.y;

	if ((p < np) && (q < nq))
	{
		sum[r * (nq * np) + q * np + p] = (DATA_TYPE)0.0;
	
		for (int s = 0; s < np; s++)
		{
			ANN_SUM DATA_TYPE tmp = A[r * (nq * np) + q * np + s] * C4[s * np + p];
			sum[r * (nq * np) + q * np + p] = sum[r * (nq * np) + q * np + p] + tmp;
		}
	}
}

extern "C" __global__ void doitgen_kernel2(int nr, int nq, int np, ANN_SUM DATA_TYPE *sum, ANN_A DATA_TYPE *A, ANN_C4 DATA_TYPE *C4, int r)
{
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	int q = blockIdx.y * blockDim.y + threadIdx.y;

	if ((p < np) && (q < nq))
	{
		A[r * (nq * np) + q * np + p] = sum[r * (nq * np) + q * np + p];
	}
}