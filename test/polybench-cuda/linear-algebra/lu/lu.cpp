/**
 * lu.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include <cuda.h>
#include <builtin_types.h>
#include <helper_cuda_drvapi.h>
#include <helper_image.h>
#include <helper_string.h>
#include <helper_timer.h>

#define POLYBENCH_TIME 1

#include "lu.cuh"
#include "lu_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
	#define PTX_FILE "lu.ptx"
#else
	#define PTX_FILE "lu.taffo.ptx"
#endif
#endif

#define RUN_ON_CPU

static int initCUDA(int argc, char **argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernels[2];
size_t totalGlobalMem;

const char *sSDKsample = "PolyBench lu (Driver API)";

void lu(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
	for (int k = 0; k < _PB_N; k++)
    {
		for (int j = k + 1; j < _PB_N; j++)
		{
			A[k][j] = A[k][j] / A[k][k];
		}

		for (int i = k + 1; i < _PB_N; i++)
		{
			for (int j = k + 1; j < _PB_N; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
		}
    }
}


double frand(void)
{
	return (double)rand() / (double)RAND_MAX;
}

void init_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
	int i;
	int j;
 	POLYBENCH_2D_ARRAY_DECL(A_tmp,DATA_TYPE,N,N,n,n);

	for (i = 0; i < n; i++) {
		for (j = 0; j <= i; j++) {
    	    A_tmp[i][j] = (DATA_TYPE)(-j % n) / (n + 1);
    	}
		for (j = i + 1; j < n; j++) {
			A_tmp[i][j] = 0;
		}
		A_tmp[i][i] = 1;
	}

	/* Make the matrix positive semi-definite. */
	/* not necessary for LU, but using same code as cholesky */
	int r, s, t;
	POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
	for (r = 0; r < n; ++r)
		for (s = 0; s < n; ++s)
			(POLYBENCH_ARRAY(B))[r][s] = 0;
	for (t = 0; t < n; ++t)
		for (r = 0; r < n; ++r)
			for (s = 0; s < n; ++s) {
				(POLYBENCH_ARRAY(B))[r][s] += A_tmp[r][t] * A_tmp[s][t];
			}
	for (r = 0; r < n; ++r)
		for (s = 0; s < n; ++s)
			A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
	POLYBENCH_FREE_ARRAY(B);
}


void compareResults(int n, DATA_TYPE POLYBENCH_2D(A_cpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,N,N,n,n))
{
	int i, j, fail;
	fail = 0;
	
	// Compare a and b
	for (i=0; i<n; i++) 
	{
		for (j=0; j<n; j++) 
		{
			if (percentDiff(A_cpu[i][j], A_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void luCuda(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,N,N,n,n))
{
	CUdeviceptr AGpu;

	checkCudaErrors(cuMemAlloc(&AGpu, sizeof(DATA_TYPE) * N * N));

	checkCudaErrors(cuMemcpyHtoD(AGpu, A, sizeof(DATA_TYPE) * N * N));

	dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
	dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
	dim3 grid1(1, 1, 1);
	dim3 grid2(1, 1, 1);

	/* Start timer. */
  	polybench_start_instruments;

	for (int k = 0; k < N -1; k++)
	{
		grid1.x = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)block1.x)));
		void *args1[3] = {&n, &AGpu, &k};
		checkCudaErrors(cuLaunchKernel(
        	kernels[0], grid1.x, grid1.y, grid1.z, block1.x, block1.y, block1.z,
       		0, NULL, args1, NULL));
		checkCudaErrors(cuCtxSynchronize());

		grid2.x = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)block2.x)));
		grid2.y = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)block2.y)));
		void *args2[3] = {&n, &AGpu, &k};
		checkCudaErrors(cuLaunchKernel(
        	kernels[1], grid2.x, grid2.y, grid2.z, block2.x, block2.y, block2.z,
        	0, NULL, args2, NULL));
		checkCudaErrors(cuCtxSynchronize());
	}
	
	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	checkCudaErrors(cuMemcpyDtoH(A_outputFromGpu, AGpu, sizeof(DATA_TYPE) * N * N));

	checkCudaErrors(cuMemFree(AGpu));
  	checkCudaErrors(cuCtxDestroy(cuContext));
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * n + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}
	

int main(int argc, char *argv[])
{
	int n = N;

	ANN_A POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,N,n,n);
  	ANN_A POLYBENCH_2D_ARRAY_DECL(A_outputFromGpu,DATA_TYPE,N,N,n,n);

	init_array(n, POLYBENCH_ARRAY(A));

	initCUDA(argc, argv);

	checkCudaErrors(cuModuleGetFunction(&(kernels[0]), cuModule, "lu_kernel1"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[1]), cuModule, "lu_kernel2"));

	luCuda(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));
	

	/* Start timer. */
	polybench_start_instruments;

	//lu(n, POLYBENCH_ARRAY(A));

	/* Stop and print timer. */
	printf("CPU Time in seconds:\n");
	polybench_stop_instruments;
	polybench_print_instruments;
	//compareResults(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));


	print_array(n, POLYBENCH_ARRAY(A_outputFromGpu));

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(A_outputFromGpu);

   	return 0;
}


static int initCUDA(int argc, char **argv) {
  CUfunction cuFunction = 0;
  int major = 0, minor = 0;
  char deviceName[100];

  cuDevice = findCudaDeviceDRV(argc, (const char **)argv);

  // get compute capabilities and the devicename
  checkCudaErrors(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  checkCudaErrors(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
  checkCudaErrors(cuDeviceGetName(deviceName, sizeof(deviceName), cuDevice));
  printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

  checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, cuDevice));
  printf("  Total amount of global memory:     %llu bytes\n",
         (long long unsigned int)totalGlobalMem);

  checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

  // Create module from binary file (PTX)
  checkCudaErrors(cuModuleLoad(&cuModule, PTX_FILE));

  return 0;
}

#include <polybench.c>