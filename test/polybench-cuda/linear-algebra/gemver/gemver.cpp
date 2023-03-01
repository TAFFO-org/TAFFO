/**
 * gemver.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "gemver.cuh"
#include "gemver_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
	#define PTX_FILE "gemver.ptx"
#else
	#define PTX_FILE "gemver.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char **argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernels[3];
size_t totalGlobalMem;

const char *sSDKsample = "PolyBench gemver (Driver API)";

void gemver(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_1D(u1, N, n), DATA_TYPE POLYBENCH_1D(v1, N, n), 
	DATA_TYPE POLYBENCH_1D(u2, N, n), DATA_TYPE POLYBENCH_1D(v2, N, n), DATA_TYPE POLYBENCH_1D(w, N, n), DATA_TYPE POLYBENCH_1D(x, N, n), DATA_TYPE POLYBENCH_1D(y, N, n), 
	DATA_TYPE POLYBENCH_1D(z, N, n))
{
	int i,j;
	
  	for (i = 0; i < _PB_N; i++)
	{
    	for (j = 0; j < _PB_N; j++)
		{
      		A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
		}
	}

  	for (i = 0; i < _PB_N; i++)
	{
    	for (j = 0; j < _PB_N; j++)
		{
      		x[i] = x[i] + beta * A[j][i] * y[j];
		}
	}

  	for (i = 0; i < _PB_N; i++)
	{
    	x[i] = x[i] + z[i];
	}

  	for (i = 0; i < _PB_N; i++)
	{
    	for (j = 0; j < _PB_N; j++)
		{
      		w[i] = w[i] +  alpha * A[i][j] * x[j];
		}
	}
}


void init(int n, DATA_TYPE *alpha,
	DATA_TYPE *beta,
	DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
	DATA_TYPE POLYBENCH_1D(u1,N,n),
	DATA_TYPE POLYBENCH_1D(v1,N,n),
	DATA_TYPE POLYBENCH_1D(u2,N,n),
	DATA_TYPE POLYBENCH_1D(v2,N,n),
	DATA_TYPE POLYBENCH_1D(w,N,n),
	DATA_TYPE POLYBENCH_1D(x,N,n),
	DATA_TYPE POLYBENCH_1D(y,N,n),
	DATA_TYPE POLYBENCH_1D(z,N,n))
{
	__attribute__((annotate("scalar(range(0, 10000) final)"))) int i;
	__attribute__((annotate("scalar(range(0, 10000) final)"))) int j;

	*alpha = 1.5;
	*beta = 1.2;

  	for (i = 0; i < N; i++)
	{
	    u1[i] = i / 2.0;
	    u2[i] = (i+1)/N/2.0;
	    v1[i] = (i+1)/N/4.0;
	    v2[i] = (i+1)/N/6.0;
	    y[i] = (i+1)/N/16.0;
	    z[i] = (i+1)/N/18.0;
	    x[i] = 0.0;
	    w[i] = 0.0;

    	for (j = 0; j < N; j++)
		{
			A[i][j] = (DATA_TYPE) (i*j % N)  / N;
		}
	}
}


void compareResults(int n, DATA_TYPE POLYBENCH_1D(w1, N, n), DATA_TYPE POLYBENCH_1D(w2, N, n))
{
	int i, fail;
	fail = 0;
	
	for (i=0; i < N; i++) 
	{
		if (percentDiff(w1[i], w2[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}
		
	// Print results
	printf("Number of misses: %d\n", fail);
}


void gemverCuda(int n, DATA_TYPE alpha, DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(u1,N,n),
		DATA_TYPE POLYBENCH_1D(v1,N,n),
		DATA_TYPE POLYBENCH_1D(u2,N,n),
		DATA_TYPE POLYBENCH_1D(v2,N,n),
		DATA_TYPE POLYBENCH_1D(w,N,n),
		DATA_TYPE POLYBENCH_1D(w_outputFromGpu,N,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(y,N,n),
		DATA_TYPE POLYBENCH_1D(z,N,n))
{
	CUdeviceptr A_gpu;
	CUdeviceptr x_gpu;
	CUdeviceptr y_gpu;
	CUdeviceptr z_gpu;
	CUdeviceptr v1_gpu;
	CUdeviceptr v2_gpu;
	CUdeviceptr u1_gpu;
	CUdeviceptr u2_gpu;
	CUdeviceptr w_gpu;

	DATA_TYPE ANN_ALPHA alpha_l[1] = {alpha};
	DATA_TYPE ANN_BETA beta_l[1] = {beta};

	checkCudaErrors(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * N * N));
	checkCudaErrors(cuMemAlloc(&x_gpu, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemAlloc(&y_gpu, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemAlloc(&z_gpu, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemAlloc(&w_gpu, sizeof(DATA_TYPE) * N ));
	checkCudaErrors(cuMemAlloc(&v1_gpu, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemAlloc(&v2_gpu, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemAlloc(&u1_gpu, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemAlloc(&u2_gpu, sizeof(DATA_TYPE) * N));
	
	checkCudaErrors(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * N * N));
	checkCudaErrors(cuMemcpyHtoD(x_gpu, x, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemcpyHtoD(y_gpu, y, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemcpyHtoD(z_gpu, z, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemcpyHtoD(w_gpu, w, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemcpyHtoD(v1_gpu, v1, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemcpyHtoD(v2_gpu, v2, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemcpyHtoD(u1_gpu, u1, sizeof(DATA_TYPE) * N));
	checkCudaErrors(cuMemcpyHtoD(u2_gpu, u2, sizeof(DATA_TYPE) * N));

	dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
	dim3 grid1((size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)), (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_1_Y)));

	dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
	dim3 grid2((size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)), 1);
	
	dim3 block3(DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y);
	dim3 grid3((size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)), 1);
	
 	/* Start timer. */
  	polybench_start_instruments;

	void *args1[8] = {&n, &alpha_l, &beta_l, &A_gpu, &v1_gpu, &v2_gpu, &u1_gpu, &u2_gpu};
	checkCudaErrors(cuLaunchKernel(
        kernels[0], grid1.x, grid1.y, grid1.z, block1.x, block1.y, block1.z,
        0, NULL, args1, NULL));
	checkCudaErrors(cuCtxSynchronize());

	void *args2[7] = {&n, &alpha_l, &beta_l, &A_gpu, &x_gpu, &y_gpu, &z_gpu};
	checkCudaErrors(cuLaunchKernel(
        kernels[1], grid2.x, grid2.y, grid2.z, block2.x, block2.y, block2.z,
        0, NULL, args2, NULL));
	checkCudaErrors(cuCtxSynchronize());

	void *args3[6] = {&n, &alpha_l, &beta_l, &A_gpu, &x_gpu, &w_gpu};
	checkCudaErrors(cuLaunchKernel(
        kernels[2], grid3.x, grid3.y, grid3.z, block3.x, block3.y, block3.z,
        0, NULL, args3, NULL));
	checkCudaErrors(cuCtxSynchronize());

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	checkCudaErrors(cuMemcpyDtoH(w_outputFromGpu, w_gpu, sizeof(DATA_TYPE) * N));
	
	checkCudaErrors(cuMemFree(A_gpu));
  	checkCudaErrors(cuMemFree(x_gpu));
  	checkCudaErrors(cuMemFree(y_gpu));
	checkCudaErrors(cuMemFree(z_gpu));
	checkCudaErrors(cuMemFree(w_gpu));
	checkCudaErrors(cuMemFree(v1_gpu));
  	checkCudaErrors(cuMemFree(v2_gpu));
  	checkCudaErrors(cuMemFree(u1_gpu));
	checkCudaErrors(cuMemFree(u2_gpu));
  	checkCudaErrors(cuCtxDestroy(cuContext));
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  for (i = 0; i < n; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, w[i]);
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
	ANN_U1 POLYBENCH_1D_ARRAY_DECL(u1,DATA_TYPE,N,n);
  	ANN_V1 POLYBENCH_1D_ARRAY_DECL(v1,DATA_TYPE,N,n);
  	ANN_U2 POLYBENCH_1D_ARRAY_DECL(u2,DATA_TYPE,N,n);
  	ANN_V2 POLYBENCH_1D_ARRAY_DECL(v2,DATA_TYPE,N,n);
  	ANN_W POLYBENCH_1D_ARRAY_DECL(w,DATA_TYPE,N,n);
  	ANN_W POLYBENCH_1D_ARRAY_DECL(w_outputFromGpu,DATA_TYPE,N,n);
  	ANN_X POLYBENCH_1D_ARRAY_DECL(x,DATA_TYPE,N,n);
  	ANN_Y POLYBENCH_1D_ARRAY_DECL(y,DATA_TYPE,N,n);
  	ANN_Z POLYBENCH_1D_ARRAY_DECL(z,DATA_TYPE,N,n);
  	
	
	init(n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(u1),
	      POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2),
	      POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
	      POLYBENCH_ARRAY(z));

	initCUDA(argc, argv);

	checkCudaErrors(cuModuleGetFunction(&(kernels[0]), cuModule, "gemver_kernel1"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[1]), cuModule, "gemver_kernel2"));	
	checkCudaErrors(cuModuleGetFunction(&(kernels[2]), cuModule, "gemver_kernel3"));	

	gemverCuda(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2), POLYBENCH_ARRAY(v2), 
		POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(w_outputFromGpu), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));

	/* Start timer. */
	polybench_start_instruments;
	//gemver(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2), POLYBENCH_ARRAY(v2), 
	//	POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));


	/* Stop and print timer. */
	printf("CPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;
	
	//compareResults(n, POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(w_outputFromGpu));

	print_array(n, POLYBENCH_ARRAY(w_outputFromGpu));

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(w);  
	POLYBENCH_FREE_ARRAY(w_outputFromGpu);  
	POLYBENCH_FREE_ARRAY(x);  
	POLYBENCH_FREE_ARRAY(y);
	POLYBENCH_FREE_ARRAY(z);
	POLYBENCH_FREE_ARRAY(u1);
	POLYBENCH_FREE_ARRAY(u2);
	POLYBENCH_FREE_ARRAY(v1);
	POLYBENCH_FREE_ARRAY(v2);

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