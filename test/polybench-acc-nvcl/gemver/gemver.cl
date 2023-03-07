/**
 * gemver.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "gemver_sh_ann.h"
typedef float DATA_TYPE;


	
__kernel void gemver_kernel1(ANN_A __global DATA_TYPE *A, ANN_V1 __global DATA_TYPE *V1, ANN_V2 __global DATA_TYPE *V2, ANN_U1 __global DATA_TYPE *U1, ANN_U2 __global DATA_TYPE *U2, int n) 
{    
	int j = get_global_id(0);
	int i = get_global_id(1);
	
	if ((i < n) && (j < n))
	{
		A[i*n + j] += U1[i] * V1[j] + U2[i] * V2[j];
	}
}


__kernel void gemver_kernel2(ANN_A __global DATA_TYPE *A, ANN_X __global DATA_TYPE *X, ANN_Y __global DATA_TYPE *Y, ANN_Z __global DATA_TYPE *Z, ANN_BETA DATA_TYPE beta, int n) 
{    
	int i = get_global_id(0);

	if (i < n)
	{
		int j;
		for(j = 0; j < n; j++) 
		{
      X[i] += beta * A[j * n + i] * Y[j];
		}
		X[i] += Z[i];
	}
}


__kernel void gemver_kernel3(ANN_A __global DATA_TYPE *A, ANN_X __global DATA_TYPE *X, ANN_W __global DATA_TYPE *w, ANN_ALPHA DATA_TYPE alpha, int n) 
{    
	int i = get_global_id(0);
	
	if (i < n)
	{
		int j;
		for(j = 0; j < n; j++)
		{ 
			w[i] += alpha * A[i*n + j] * X[j];
		}
    w[i] = X[i];
	}
}
