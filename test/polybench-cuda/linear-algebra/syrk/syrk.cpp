/**
 * syrk.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

#include <cuda.h>
#include <builtin_types.h>
#include <helper_cuda_drvapi.h>
#include <helper_image.h>
#include <helper_string.h>
#include <helper_timer.h>

#define POLYBENCH_TIME 1

#include "syrk.cuh"
#include "syrk_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
	#define PTX_FILE "syrk.ptx"
#else
	#define PTX_FILE "syrk.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char **argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernel;
size_t totalGlobalMem;

const char *sSDKsample = "PolyBench syrk (Driver API)";


void init_arrays(int ni, int nj,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,NI,NI,ni,ni),
		DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj))
{
	int i, j;
  float tmp;

	*alpha = 32412;
	*beta = 2123;
	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nj; j++)
		{
      tmp = ((DATA_TYPE) i*j) / (ni*nj);
			A[i][j] = tmp;
		}
	}

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < ni; j++)
		{
      tmp = ((DATA_TYPE) i*j) / (ni*nj);
			C[i][j] = tmp;
		}
	}
}


void syrk(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni))
{
	int i, j, k;
	
	/*  C := alpha*A*A' + beta*C */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NI; j++)
		{
			C[i][j] *= beta;
		}
	}
	
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NI; j++)
		{
			for (k = 0; k < _PB_NJ; k++)
			{
				C[i][j] += alpha * A[i][k] * A[j][k];
			}
		}
	}
}


void compareResults(int ni, DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni), DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NI, ni, ni))
{
	int i,j,fail;
	fail = 0;

	// Compare C with D
	for (i=0; i<ni; i++)
	{
		for (j=0; j<ni; j++)
		{
			if (percentDiff(C[i][j], C_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void syrkCuda(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni), 
		DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NI, ni, ni))
{
	CUdeviceptr A_gpu;
	CUdeviceptr C_gpu;

	DATA_TYPE ANN_ALPHA alpha_l[1] = {alpha};
	DATA_TYPE ANN_BETA beta_l[1] = {beta};

	checkCudaErrors(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NI * NJ));
	checkCudaErrors(cuMemAlloc(&C_gpu, sizeof(DATA_TYPE) * NI * NI));

	checkCudaErrors(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ));
	checkCudaErrors(cuMemcpyHtoD(C_gpu, C, sizeof(DATA_TYPE) * NI * NI));
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil(((float)NI) / ((float)DIM_THREAD_BLOCK_X))), (size_t)ceil(((float)NI) / ((float)DIM_THREAD_BLOCK_Y)));

	/* Start timer. */
  	polybench_start_instruments;

	void *args1[6] = {&ni, &nj, &alpha_l, &beta_l, &A_gpu, &C_gpu};
	checkCudaErrors(cuLaunchKernel(
        kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z,
        0, NULL, args1, NULL));
	checkCudaErrors(cuCtxSynchronize());


	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	checkCudaErrors(cuMemcpyDtoH(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NI));
	
	checkCudaErrors(cuMemFree(A_gpu));
	checkCudaErrors(cuMemFree(C_gpu));
  	checkCudaErrors(cuCtxDestroy(cuContext));
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, DATA_TYPE POLYBENCH_2D(C,NI,NI,ni,ni))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < ni; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, C[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	/* Retrieve problem size. */
	int ni = NI;
	int nj = NJ;

	/* Variable declaration/allocation. */
	DATA_TYPE alpha;
	DATA_TYPE beta;

	ANN_A POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,ni,nj);
  	ANN_C POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NI,ni,ni);
  	ANN_C POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu,DATA_TYPE,NI,NI,ni,ni);

	init_arrays(ni, nj, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));
	
	initCUDA(argc, argv);

	checkCudaErrors(cuModuleGetFunction(&(kernel), cuModule, "syrk_kernel"));

	syrkCuda(ni, nj, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

	/* Start timer. */
	polybench_start_instruments;

	//syrk(ni, nj, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C));

	/* Stop and print timer. */
	printf("CPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	//compareResults(ni, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

	print_array(ni, POLYBENCH_ARRAY(C_outputFromGpu));

	POLYBENCH_FREE_ARRAY(A);
  	POLYBENCH_FREE_ARRAY(C);
	POLYBENCH_FREE_ARRAY(C_outputFromGpu);

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