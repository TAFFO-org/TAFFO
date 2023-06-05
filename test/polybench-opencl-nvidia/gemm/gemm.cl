/**
 * gemm.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "gemm_sh_ann.h"
typedef float DATA_TYPE;


	
__kernel void gemm(ANN_A __global DATA_TYPE *a, ANN_B __global DATA_TYPE *b, ANN_C __global DATA_TYPE *c, ANN_ALPHA DATA_TYPE alpha, ANN_BETA DATA_TYPE beta, int ni, int nj, int nk) 
{
    	int j = get_global_id(0);
	int i = get_global_id(1);

	if ((i < ni) && (j < nj))
	{	
		c[i * nj + j] *= beta;
		int k;
		for(k=0; k < nk; k++)
		{
			c[i * nj + j] += alpha * a[i * nk + k] * b[k * nj +j];
		}
	}
}

