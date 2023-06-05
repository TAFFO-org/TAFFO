/**
 * lu.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#include "lu_sh_ann.h"

typedef float DATA_TYPE;

__kernel void lu_kernel1(ANN_A __global DATA_TYPE *A, int k, int n)
{
	int j = get_global_id(0) + (k + 1);
	
	if ((j < n))
	{
		A[k*n + j] = A[k*n + j] / A[k*n + k];
	}
}

__kernel void lu_kernel2(ANN_A __global DATA_TYPE *A, int k, int n)
{
	int j = get_global_id(0) + (k + 1);
	int i = get_global_id(1) + (k + 1);
	
	if ((i < n) && (j < n))
	{
		__attribute__((annotate("scalar()"))) DATA_TYPE tmp = A[i*n + k] * A[k*n + j];
		A[i*n + j] = A[i*n + j] - tmp;
	}
}
