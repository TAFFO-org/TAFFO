/**
 * bicg.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <sys/time.h>

#include <cuda.h>
#include <builtin_types.h>
#include <helper_cuda_drvapi.h>
#include <helper_image.h>
#include <helper_string.h>
#include <helper_timer.h>

#define POLYBENCH_TIME 1

#include "bicg.cuh"
#include "bicg_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
	#define PTX_FILE "bicg.ptx"
#else
	#define PTX_FILE "bicg.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char **argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernels[2];
size_t totalGlobalMem;

const char *sSDKsample = "PolyBench bicg (Driver API)";

void init_array(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx))
{
	__attribute__((annotate("scalar(range(0, 10000) final)"))) int i;
	__attribute__((annotate("scalar(range(0, 10000) final)"))) int j;

	for (i = 0; i < ny; i++)
	{
    	p[i] = (DATA_TYPE) (i % ny) /ny;
	}

	for (i = 0; i < nx; i++)
	{
    	r[i] = (DATA_TYPE) (i % nx) /nx;
    	for (j = 0; j < ny; j++)
		{
      		A[i][j] = (DATA_TYPE)( i*j % nx) / NX;
		}
 	}
}


void compareResults(int nx, int ny, DATA_TYPE POLYBENCH_1D(s,NY,ny), DATA_TYPE POLYBENCH_1D(s_outputFromGpu,NY,ny), 
		DATA_TYPE POLYBENCH_1D(q,NX,nx), DATA_TYPE POLYBENCH_1D(q_outputFromGpu,NX,nx))
{
	int i,fail;
	fail = 0;

	// Compare s with s_cuda
	for (i=0; i<nx; i++)
	{
		if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}

	for (i=0; i<ny; i++)
	{
		if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}		
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void bicg_cpu(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx), DATA_TYPE POLYBENCH_1D(s,NY,ny), 
		DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(q,NX,nx))
{
	int i,j;
	
	ANN_Q DATA_TYPE max_q = 0;
	ANN_S DATA_TYPE max_s = 0;
  	for (i = 0; i < _PB_NY; i++)
	{
		s[i] = 0.0;
	}

	for (i = 0; i < _PB_NX; i++)
	{
		q[i] = 0.0;
		for (j = 0; j < _PB_NY; j++)
	  	{
	    	s[j] = s[j] + r[i] * A[i][j];
	    	q[i] = q[i] + A[i][j] * p[j];
			if( s[j] > max_s)
				max_s = s[j];
			if(q[i] > max_q)
				max_q = q[i];	
	  	}
	}
	//printf("max q and s: %lf %lf\n", max_q, max_s);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx, int ny,
		 DATA_TYPE POLYBENCH_1D(s,NY,ny),
		 DATA_TYPE POLYBENCH_1D(q,NX,nx))

{
  int i;

  for (i = 0; i < ny; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, s[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  for (i = 0; i < nx; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, q[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");
}


void bicgCuda(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx), DATA_TYPE POLYBENCH_1D(s,NY,ny), 
	DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(q,NX,nx), DATA_TYPE POLYBENCH_1D(s_outputFromGpu,NY,ny), 
	DATA_TYPE POLYBENCH_1D(q_outputFromGpu,NX,nx))
{
	CUdeviceptr A_gpu;
	CUdeviceptr q_gpu;
	CUdeviceptr p_gpu;
	CUdeviceptr r_gpu;
	CUdeviceptr s_gpu;

	checkCudaErrors(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NX * NY));
	checkCudaErrors(cuMemAlloc(&r_gpu, sizeof(DATA_TYPE) * NX));
	checkCudaErrors(cuMemAlloc(&s_gpu, sizeof(DATA_TYPE) * NY));
	checkCudaErrors(cuMemAlloc(&p_gpu, sizeof(DATA_TYPE) * NY));
	checkCudaErrors(cuMemAlloc(&q_gpu, sizeof(DATA_TYPE) * NX));

	checkCudaErrors(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NX * NY));
	checkCudaErrors(cuMemcpyHtoD(r_gpu, r, sizeof(DATA_TYPE) * NX));
	checkCudaErrors(cuMemcpyHtoD(s_gpu, s, sizeof(DATA_TYPE) * NY));
	checkCudaErrors(cuMemcpyHtoD(p_gpu, p, sizeof(DATA_TYPE) * NY));
	checkCudaErrors(cuMemcpyHtoD(q_gpu, q, sizeof(DATA_TYPE) * NX));

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);

	/* Start timer. */
  	polybench_start_instruments;

	void *args1[5] = {&nx, &ny, &A_gpu, &r_gpu, &s_gpu};
	checkCudaErrors(cuLaunchKernel(
        kernels[0], grid1.x, grid1.y, grid1.z, block.x, block.y, block.z,
        0, NULL, args1, NULL));
	checkCudaErrors(cuCtxSynchronize());

	void *args2[5] = {&nx, &ny, &A_gpu, &p_gpu, &q_gpu};
	checkCudaErrors(cuLaunchKernel(
        kernels[1], grid2.x, grid2.y, grid2.z, block.x, block.y, block.z,
        0, NULL, args2, NULL));
	checkCudaErrors(cuCtxSynchronize());

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;
	
	checkCudaErrors(cuMemcpyDtoH(s_outputFromGpu, s_gpu, sizeof(DATA_TYPE) * NY));
	checkCudaErrors(cuMemcpyDtoH(q_outputFromGpu, q_gpu, sizeof(DATA_TYPE) * NX));

	checkCudaErrors(cuMemFree(A_gpu));
  	checkCudaErrors(cuMemFree(r_gpu));
  	checkCudaErrors(cuMemFree(s_gpu));
	checkCudaErrors(cuMemFree(p_gpu));
	checkCudaErrors(cuMemFree(q_gpu));
  	checkCudaErrors(cuCtxDestroy(cuContext));
}


int main(int argc, char** argv)
{
	int nx = NX;
	int ny = NY;

	ANN_A POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NX,NY,nx,ny);
	ANN_S POLYBENCH_1D_ARRAY_DECL(s,DATA_TYPE,NY,ny);
	ANN_Q POLYBENCH_1D_ARRAY_DECL(q,DATA_TYPE,NX,nx);
	ANN_P POLYBENCH_1D_ARRAY_DECL(p,DATA_TYPE,NY,ny);
	ANN_R POLYBENCH_1D_ARRAY_DECL(r,DATA_TYPE,NX,nx);
	ANN_S POLYBENCH_1D_ARRAY_DECL(s_outputFromGpu,DATA_TYPE,NY,ny);
	ANN_Q POLYBENCH_1D_ARRAY_DECL(q_outputFromGpu,DATA_TYPE,NX,nx);

	init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(r));

	initCUDA(argc, argv);

	checkCudaErrors(cuModuleGetFunction(&(kernels[0]), cuModule, "bicg_kernel1"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[1]), cuModule, "bicg_kernel2"));

	bicgCuda(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q), 
		POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q_outputFromGpu));


	/* Start timer. */
	polybench_start_instruments;

	//bicg_cpu(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

	/* Stop and print timer. */
	printf("CPU Time in seconds:\n");
	polybench_stop_instruments;
	polybench_print_instruments;

	//compareResults(nx, ny, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q), 
	//	POLYBENCH_ARRAY(q_outputFromGpu));


	print_array(nx, ny, POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q_outputFromGpu));


	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(r);
	POLYBENCH_FREE_ARRAY(s);
	POLYBENCH_FREE_ARRAY(p);
	POLYBENCH_FREE_ARRAY(q);
	POLYBENCH_FREE_ARRAY(s_outputFromGpu);
	POLYBENCH_FREE_ARRAY(q_outputFromGpu);

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