/**
 * jacobi1D.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "jacobi1D_sh_ann.h"

typedef float DATA_TYPE;


__kernel void runJacobi1D_kernel1(ANN_A __global DATA_TYPE* A, ANN_B __global DATA_TYPE* B, int n)
{
	int i = get_global_id(0);
	if ((i >= 1) && (i < (n-1)))
	{
		B[i] = (A[i-1] + A[i] + A[i + 1]) / 3.0f;
	}
}

__kernel void runJacobi1D_kernel2(ANN_A __global DATA_TYPE* A, ANN_B __global DATA_TYPE* B, int n)
{
	int j = get_global_id(0);
	
	if ((j >= 1) && (j < (n-1)))
	{
		A[j] = B[j];
	}
}
