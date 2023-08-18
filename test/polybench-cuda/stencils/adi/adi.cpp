/**
 * adi.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "adi.cuh"
#include "adi_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 2.5

#define GPU_DEVICE 0

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
	#define PTX_FILE "adi.ptx"
#else
	#define PTX_FILE "adi.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char **argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernels[6];
size_t totalGlobalMem;

const char *sSDKsample = "PolyBench adi (Driver API)";

void adi(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n))
{	
	for (int t = 0; t < _PB_TSTEPS; t++)
    {
    	for (int i1 = 0; i1 < _PB_N; i1++)
		{
			for (int i2 = 1; i2 < _PB_N; i2++)
			{
				X[i1][i2] = X[i1][i2] - X[i1][(i2-1)] * A[i1][i2] / B[i1][(i2-1)];
				B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][(i2-1)];		
			}
		}
	   	for (int i1 = 0; i1 < _PB_N; i1++)
		{
			X[i1][(N-1)] = X[i1][(N-1)] / B[i1][(N-1)];
		}

	   	for (int i1 = 0; i1 < _PB_N; i1++)
		{
			for (int i2 = 0; i2 < _PB_N-2; i2++)
			{
				X[i1][(N-i2-2)] = (X[i1][(N-2-i2)] - X[i1][(N-2-i2-1)] * A[i1][(N-i2-3)]) / B[i1][(N-3-i2)];	
			}
		}

	   	for (int i1 = 1; i1 < _PB_N; i1++)
		{
			for (int i2 = 0; i2 < _PB_N; i2++) 
			{
		  		X[i1][i2] = X[i1][i2] - X[(i1-1)][i2] * A[i1][i2] / B[(i1-1)][i2];
		  		B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[(i1-1)][i2];
			}
		}

	   	for (int i2 = 0; i2 < _PB_N; i2++)
		{
			X[(N-1)][i2] = X[(N-1)][i2] / B[(N-1)][i2];
		}

	   	for (int i1 = 0; i1 < _PB_N-2; i1++)
		{
			for (int i2 = 0; i2 < _PB_N; i2++)
			{
		 	 	X[(N-2-i1)][i2] = (X[(N-2-i1)][i2] - X[(N-i1-3)][i2] * A[(N-3-i1)][i2]) / B[(N-2-i1)][i2];					
			}
		}
    }
}


void init_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n))
{
  	int i, j;
    DATA_TYPE tmp;

  	for (i = 0; i < n; i++)
	{
    		for (j = 0; j < n; j++)
      		{
            tmp = (((DATA_TYPE) i*(j+1) + 1) / (N*N*4))+0.75;
			X[i][j] = tmp;
      tmp = (((DATA_TYPE) (i-1)*(j+4) + 2) / (N*N*4));
			A[i][j] = tmp;
      tmp = (((DATA_TYPE) (i+3)*(j+7) + 3) / (N*N)) + 1;
			B[i][j] = tmp;
      		}
	}
}


void compareResults(int n, DATA_TYPE POLYBENCH_2D(B_cpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(B_fromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(X_cpu,N,N,n,n), 
			DATA_TYPE POLYBENCH_2D(X_fromGpu,N,N,n,n))
{
	int i, j, fail;
	fail = 0;
	
	// Compare b and x output on cpu and gpu
	for (i=0; i < n; i++) 
	{
		for (j=0; j < n; j++) 
		{
			if (percentDiff(B_cpu[i][j], B_fromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	for (i=0; i<n; i++) 
	{
		for (j=0; j<n; j++) 
		{
			if (percentDiff(X_cpu[i][j], X_fromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}

	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void adiCuda(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n), 
	DATA_TYPE POLYBENCH_2D(B_outputFromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(X_outputFromGpu,N,N,n,n))
{
	CUdeviceptr A_gpu;
	CUdeviceptr B_gpu;
	CUdeviceptr X_gpu;

	checkCudaErrors(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * N * N));
	checkCudaErrors(cuMemAlloc(&B_gpu, sizeof(DATA_TYPE) * N * N));
	checkCudaErrors(cuMemAlloc(&X_gpu, sizeof(DATA_TYPE) * N * N));

	checkCudaErrors(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * N * N));
	checkCudaErrors(cuMemcpyHtoD(B_gpu, B, sizeof(DATA_TYPE) * N * N));
	checkCudaErrors(cuMemcpyHtoD(X_gpu, X, sizeof(DATA_TYPE) * N * N));

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1);
	dim3 grid(1, 1, 1);
	grid.x = (size_t)(ceil( ((float)N) / ((float)block.x) ));

	/* Start timer. */
  	polybench_start_instruments;

	for (int t = 0; t < _PB_TSTEPS; t++)
	{
		void *args1[4] = {&n, &A_gpu, &B_gpu, &X_gpu};
		checkCudaErrors(cuLaunchKernel(
    	    kernels[0], grid.x, grid.y, grid.z, block.x, block.y, block.z,
    	    0, NULL, args1, NULL));
		checkCudaErrors(cuCtxSynchronize());

		checkCudaErrors(cuLaunchKernel(
    	    kernels[1], grid.x, grid.y, grid.z, block.x, block.y, block.z,
    	    0, NULL, args1, NULL));
		checkCudaErrors(cuCtxSynchronize());

		checkCudaErrors(cuLaunchKernel(
    	    kernels[2], grid.x, grid.y, grid.z, block.x, block.y, block.z,
    	    0, NULL, args1, NULL));
		checkCudaErrors(cuCtxSynchronize());
	
		for (int i1 = 1; i1 < _PB_N; i1++)
		{
			void *args2[5] = {&n, &A_gpu, &B_gpu, &X_gpu, &i1};
			checkCudaErrors(cuLaunchKernel(
    	    	kernels[3], grid.x, grid.y, grid.z, block.x, block.y, block.z,
    	    	0, NULL, args2, NULL));
			checkCudaErrors(cuCtxSynchronize());
		}

		checkCudaErrors(cuLaunchKernel(
    	    kernels[4], grid.x, grid.y, grid.z, block.x, block.y, block.z,
    	    0, NULL, args1, NULL));
		checkCudaErrors(cuCtxSynchronize());
		
		for (int i1 = 0; i1 < _PB_N-2; i1++)
		{
			void *args2[5] = {&n, &A_gpu, &B_gpu, &X_gpu, &i1};
			checkCudaErrors(cuLaunchKernel(
    	    	kernels[5], grid.x, grid.y, grid.z, block.x, block.y, block.z,
    	    	0, NULL, args2, NULL));
			checkCudaErrors(cuCtxSynchronize());
		}
	}

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	checkCudaErrors(cuMemcpyDtoH(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * N * N));
	checkCudaErrors(cuMemcpyDtoH(X_outputFromGpu, X_gpu, sizeof(DATA_TYPE) * N * N));

	checkCudaErrors(cuMemFree(A_gpu));
	checkCudaErrors(cuMemFree(B_gpu));
	checkCudaErrors(cuMemFree(X_gpu));
  	checkCudaErrors(cuCtxDestroy(cuContext));
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(X,N,N,n,n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, X[i][j]);
      if ((i * N + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


int main(int argc, char *argv[])
{
	int tsteps = TSTEPS;
	int n = N;

	ANN_A POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,N,n,n);
	ANN_B POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,N,n,n);
	ANN_B POLYBENCH_2D_ARRAY_DECL(B_outputFromGpu,DATA_TYPE,N,N,n,n);
	ANN_X POLYBENCH_2D_ARRAY_DECL(X,DATA_TYPE,N,N,n,n);
	ANN_X POLYBENCH_2D_ARRAY_DECL(X_outputFromGpu,DATA_TYPE,N,N,n,n);

	initCUDA(argc, argv);

	checkCudaErrors(cuModuleGetFunction(&(kernels[0]), cuModule, "adi_kernel1"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[1]), cuModule, "adi_kernel2"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[2]), cuModule, "adi_kernel3"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[3]), cuModule, "adi_kernel4"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[4]), cuModule, "adi_kernel5"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[5]), cuModule, "adi_kernel6"));

	init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(X));

	adiCuda(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(B_outputFromGpu), 
		POLYBENCH_ARRAY(X_outputFromGpu));
	
	/* Start timer. */
	polybench_start_instruments;

	//adi(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(X));

	/* Stop and print timer. */
	printf("CPU Time in seconds:\n");
	polybench_stop_instruments;
	polybench_print_instruments;

	//compareResults(n, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu), POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(X_outputFromGpu));

	print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(B_outputFromGpu);
	POLYBENCH_FREE_ARRAY(X);
	POLYBENCH_FREE_ARRAY(X_outputFromGpu);

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