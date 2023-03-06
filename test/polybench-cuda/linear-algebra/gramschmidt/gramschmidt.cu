/**
 * gramschmidt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */
#include "gramschmidt.cuh"
#include "gramschmidt_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>


extern "C"  __global__ void gramschmidt_kernel1(int ni, int nj, ANN_A DATA_TYPE *a, ANN_R DATA_TYPE *r, ANN_Q DATA_TYPE *q, int k)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid==0)
	{
		DATA_TYPE __attribute__((annotate("scalar()"))) nrm = 0.0;
		int i;
		for (i = 0; i < _PB_NI; i++)
		{
			nrm += a[i * NJ + k] * a[i * NJ + k];
		}
      		r[k * NJ + k] = sqrt(nrm);
	}
}


extern "C"  __global__ void gramschmidt_kernel2(int ni, int nj, ANN_A DATA_TYPE *a, ANN_R DATA_TYPE *r, ANN_Q DATA_TYPE *q, int k)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < _PB_NI)
	{	
		q[i * NJ + k] = a[i * NJ + k] / r[k * NJ + k];
	}
}


extern "C"  __global__ void gramschmidt_kernel3(int ni, int nj, ANN_A DATA_TYPE *a, ANN_R DATA_TYPE *r, ANN_Q DATA_TYPE *q, int k)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((j > k) && (j < _PB_NJ))
	{
		r[k*NJ + j] = 0.0;

		int i;
		for (i = 0; i < _PB_NI; i++)
		{
			r[k*NJ + j] += q[i*NJ + k] * a[i*NJ + j];
		}
		
		for (i = 0; i < _PB_NI; i++)
		{
			a[i*NJ + j] -= q[i*NJ + k] * r[k*NJ + j];
		}
	}
}