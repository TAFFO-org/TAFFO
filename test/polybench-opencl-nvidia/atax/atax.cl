/**
 * atax.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "atax_sh_ann.h"
typedef float DATA_TYPE;

__kernel void atax_kernel1(ANN_A __global DATA_TYPE *A, ANN_X __global DATA_TYPE *x, ANN_TMP __global DATA_TYPE *tmp, int nx, int ny) {
    
	int i = get_global_id(0);

	if (i < nx)
	{
		int j;
		for(j=0; j < ny; j++)
		{
			tmp[i] += A[i * ny + j] * x[j];
		}
	}
}

__kernel void atax_kernel2(ANN_A __global DATA_TYPE *A, ANN_Y __global DATA_TYPE *y, ANN_TMP __global DATA_TYPE *tmp, int nx, int ny) {
    
	int j = get_global_id(0);

	if (j < ny)
	{
		int i;
    y[j] = 0;
		for(i=0; i < nx; i++)
		{
			y[j] += A[i * ny + j] * tmp[i];
		}
	}
}

