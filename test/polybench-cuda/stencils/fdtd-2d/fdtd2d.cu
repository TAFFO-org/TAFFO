/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */
#include "fdtd2d.cuh"
#include "fdtd2d_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

extern "C" __global__ void fdtd_step1_kernel(int nx, int ny, ANN_FICT DATA_TYPE* _fict_, ANN_EX DATA_TYPE *ex, ANN_EY DATA_TYPE *ey, ANN_HZ DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_NX) && (j < _PB_NY))
	{
		if (i == 0) 
		{
			ey[i * NY + j] = _fict_[t];
		}
		else
		{ 
			ey[i * NY + j] = ey[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]);
		}
	}
}



extern "C" __global__ void fdtd_step2_kernel(int nx, int ny, ANN_EX DATA_TYPE *ex, ANN_EY DATA_TYPE *ey, ANN_HZ DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < _PB_NX) && (j < _PB_NY) && (j > 0))
	{
		ex[i * NY + j] = ex[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
	}
}


extern "C" __global__ void fdtd_step3_kernel(int nx, int ny, ANN_EX DATA_TYPE *ex, ANN_EY DATA_TYPE *ey, ANN_HZ DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < (_PB_NX-1)) && (j < (_PB_NY-1)))
	{	
		hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * NY + (j+1)] - ex[i * NY + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
	}
}