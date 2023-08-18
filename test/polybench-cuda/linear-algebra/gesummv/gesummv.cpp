/**
 * gesummv.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "gesummv.cuh"
#include "gesummv_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 43532.0f
#define BETA 12313.0f

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
	#define PTX_FILE "gesummv.ptx"
#else
	#define PTX_FILE "gesummv.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char **argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernel;
size_t totalGlobalMem;

const char *sSDKsample = "PolyBench gesummv (Driver API)";

void gesummv(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_1D(tmp,N,n),
		DATA_TYPE POLYBENCH_1D(x,N,n), DATA_TYPE POLYBENCH_1D(y,N,n))
{
	int i, j;
	
	for (i = 0; i < _PB_N; i++)
	{
		tmp[i] = 0;
		y[i] = 0;
		for (j = 0; j < _PB_N; j++)
		{
			tmp[i] = A[i][j] * x[j] + tmp[i];
			y[i] = B[i][j] * x[j] + y[i];
		}
		
		y[i] = alpha * tmp[i] + beta * y[i];
	}
}


void init(int n, DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), 
	DATA_TYPE POLYBENCH_1D(x,N,n))
{
  	int i, j;
    float t;

	*alpha = 43532;
	*beta = 12313;

 	for (i = 0; i < n; i++)
    	{
        t = ((DATA_TYPE) i) / N;
    		x[i] = t;
		for (j = 0; j < n; j++) 
		{
      t = ((DATA_TYPE) i*j) / (N*N);
			A[i][j] = t;
      t = ((DATA_TYPE) i*j) / (N*N);
			B[i][j] = t;
		}
	}
}


void compareResults(int n, DATA_TYPE POLYBENCH_1D(y,N,n), DATA_TYPE POLYBENCH_1D(y_outputFromGpu,N,n))
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<n; i++) 
	{
		if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void gesummvCuda(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), 
		DATA_TYPE POLYBENCH_1D(tmp,N,n), DATA_TYPE POLYBENCH_1D(x,N,n), DATA_TYPE POLYBENCH_1D(y,N,n),  
		DATA_TYPE POLYBENCH_1D(y_outputFromGpu,N,n))
{
	CUdeviceptr A_gpu;
	CUdeviceptr B_gpu;
	CUdeviceptr x_gpu;
	CUdeviceptr y_gpu;
	CUdeviceptr tmp_gpu;

	DATA_TYPE ANN_ALPHA alpha_l[1] = {alpha};
	DATA_TYPE ANN_BETA beta_l[1] = {beta};

	checkCudaErrors(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * N * N));
	checkCudaErrors(cuMemAlloc(&B_gpu, sizeof(DATA_TYPE) * N * N));
	checkCudaErrors(cuMemAlloc(&x_gpu, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemAlloc(&y_gpu, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemAlloc(&tmp_gpu, sizeof(DATA_TYPE) * N));
	
	checkCudaErrors(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * N * N));
	checkCudaErrors(cuMemcpyHtoD(B_gpu, B, sizeof(DATA_TYPE) * N * N));
	checkCudaErrors(cuMemcpyHtoD(x_gpu, x, sizeof(DATA_TYPE) * N));

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), 1);


	/* Start timer. */
  	polybench_start_instruments;

	void *args1[8] = {&n, &alpha_l, &beta_l, &A_gpu, &B_gpu, &tmp_gpu, &x_gpu, &y_gpu};
	checkCudaErrors(cuLaunchKernel(
        kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z,
        0, NULL, args1, NULL));
	checkCudaErrors(cuCtxSynchronize());

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	checkCudaErrors(cuMemcpyDtoH(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * N));

	checkCudaErrors(cuMemFree(A_gpu));
	checkCudaErrors(cuMemFree(B_gpu));
	checkCudaErrors(cuMemFree(x_gpu));
	checkCudaErrors(cuMemFree(y_gpu));
	checkCudaErrors(cuMemFree(tmp_gpu));
  	checkCudaErrors(cuCtxDestroy(cuContext));
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  for (i = 0; i < n; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, y[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
}


int main(int argc, char *argv[])
{
	/* Retrieve problem size. */
	int n = N;

	/* Variable declaration/allocation. */
	DATA_TYPE alpha;
	DATA_TYPE beta;
	ANN_A POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,N,n,n);
	ANN_B POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,N,n,n);
	ANN_TMP POLYBENCH_1D_ARRAY_DECL(tmp,DATA_TYPE,N,n);
	ANN_X POLYBENCH_1D_ARRAY_DECL(x,DATA_TYPE,N,n);
	ANN_Y POLYBENCH_1D_ARRAY_DECL(y,DATA_TYPE,N,n);
	ANN_Y POLYBENCH_1D_ARRAY_DECL(y_outputFromGpu,DATA_TYPE,N,n);

	init(n, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(x));
	
	initCUDA(argc, argv);

	checkCudaErrors(cuModuleGetFunction(&(kernel), cuModule, "gesummv_kernel"));

	gesummvCuda(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y),  
		POLYBENCH_ARRAY(y_outputFromGpu));

	/* Start timer. */
	polybench_start_instruments;

	//gesummv(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y));
	
	/* Stop and print timer. */
	printf("CPU Time in seconds:\n");
	polybench_stop_instruments;
	polybench_print_instruments;

	//compareResults(n, POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(y_outputFromGpu));

	print_array(n, POLYBENCH_ARRAY(y_outputFromGpu));
	//print_array(n, POLYBENCH_ARRAY(tmp));

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);  
	POLYBENCH_FREE_ARRAY(tmp);
	POLYBENCH_FREE_ARRAY(x);  
	POLYBENCH_FREE_ARRAY(y);
	POLYBENCH_FREE_ARRAY(y_outputFromGpu);

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