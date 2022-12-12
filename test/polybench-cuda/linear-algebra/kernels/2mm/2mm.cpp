/**
 * 2mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "2mm.cuh"

#include "2mm_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
	#define PTX_FILE "2mm.ptx"
#else
	#define PTX_FILE "2mm.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char **argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernels[2];
size_t totalGlobalMem;

const char *sSDKsample = "PolyBench correlation (Driver API)";

void init_array(int ni, int nj, int nk, int nl, DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk), 
		DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj), DATA_TYPE POLYBENCH_2D(C, NL, NJ, nl, nj), 
		DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl))
{
	int i, j;

	*alpha = 32412;
	*beta = 2123;

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nk; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / NI;
		}
	}

	for (i = 0; i < nk; i++)
	{
		for (j = 0; j < nj; j++)
		{
			B[i][j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}

	for (i = 0; i < nl; i++)
	{
		for (j = 0; j < nj; j++)
		{
			C[i][j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nl; j++)
		{
			D[i][j] = ((DATA_TYPE) i*(j+2)) / NK;	
		}
	}
}


void compareResults(int ni, int nl, DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl), DATA_TYPE POLYBENCH_2D(D_outputFromGpu, NI, NL, ni, nl))
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < ni; i++)
	{
		for (j=0; j < nl; j++)
		{
			DATA_TYPE a = D[i][j];
			DATA_TYPE b = D_outputFromGpu[i][j];
			if (percentDiff(a,b) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void mm2_cpu(int ni, int nj, int nk, int nl,
		DATA_TYPE alpha,
		DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
	int i, j, k;
	
	/* D := alpha*A*B*C + beta*D */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NJ; j++)
		{
			tmp[i][j] = 0;
			for (k = 0; k < _PB_NK; ++k)
			{
				tmp[i][j] += alpha * A[i][k] * B[k][j];
			}
		}
	}

	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NL; j++)
		{
			D[i][j] *= beta;
			for (k = 0; k < _PB_NJ; ++k)
			{
				D[i][j] += tmp[i][k] * C[k][j];
			}
		}
	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, D[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


void mm2Cuda(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj), 
	DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj), 
	DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl), DATA_TYPE POLYBENCH_2D(D_outputFromGpu,NI,NL,ni,nl))
{
	CUdeviceptr tmp_gpu;
	CUdeviceptr A_gpu;
	CUdeviceptr B_gpu;
	CUdeviceptr C_gpu;
	CUdeviceptr D_gpu;

	checkCudaErrors(cuMemAlloc(&tmp_gpu, sizeof(DATA_TYPE) * NI * NJ));
	checkCudaErrors(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NI * NK));
	checkCudaErrors(cuMemAlloc(&B_gpu, sizeof(DATA_TYPE) * NK * NJ));
	checkCudaErrors(cuMemAlloc(&C_gpu, sizeof(DATA_TYPE) * NL * NJ));
	checkCudaErrors(cuMemAlloc(&D_gpu, sizeof(DATA_TYPE) * NI * NL));

	checkCudaErrors(cuMemcpyHtoD(tmp_gpu, tmp, sizeof(DATA_TYPE) * NI * NJ));
	checkCudaErrors(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NI * NK));
	checkCudaErrors(cuMemcpyHtoD(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ));
	checkCudaErrors(cuMemcpyHtoD(C_gpu, C, sizeof(DATA_TYPE) * NL * NJ));
	checkCudaErrors(cuMemcpyHtoD(D_gpu, D, sizeof(DATA_TYPE) * NI * NL));
		
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)ceil( ((float)NJ) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );
	dim3 grid2((size_t)ceil( ((float)NL) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );

	/* Start timer. */
  	polybench_start_instruments;

	void *args1[9] = {&ni, &nj, &nk, &nl, &alpha, &beta, &tmp_gpu, &A_gpu, &B_gpu};
	checkCudaErrors(cuLaunchKernel(
        kernels[0], grid1.x, grid1.y, grid1.z, block.x, block.y, block.z,
        0, NULL, args1, NULL));
	checkCudaErrors(cuCtxSynchronize());

	void *args2[9] = {&ni, &nj, &nk, &nl, &alpha, &beta, &tmp_gpu, &C_gpu, &D_gpu};
	checkCudaErrors(cuLaunchKernel(
        kernels[1], grid2.x, grid2.y, grid2.z, block.x, block.y, block.z,
        0, NULL, args2, NULL));
	checkCudaErrors(cuCtxSynchronize());

	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	checkCudaErrors(cuMemcpyDtoH(D_outputFromGpu, D_gpu, sizeof(DATA_TYPE) * NI * NL));

	checkCudaErrors(cuMemFree(tmp_gpu));
  	checkCudaErrors(cuMemFree(A_gpu));
  	checkCudaErrors(cuMemFree(B_gpu));
	checkCudaErrors(cuMemFree(C_gpu));
	checkCudaErrors(cuMemFree(D_gpu));
  	checkCudaErrors(cuCtxDestroy(cuContext));
}


int main(int argc, char** argv)
{
	/* Retrieve problem size. */
	int ni = NI;
	int nj = NJ;
	int nk = NK;
	int nl = NL;

	/* Variable declaration/allocation. */
	ANN_ALPHA DATA_TYPE alpha;
	ANN_BETA DATA_TYPE beta;
	ANN_TMP POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
	ANN_A POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
	ANN_B POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
	ANN_C POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NL,NJ,nl,nj);
	ANN_D POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);
	ANN_D POLYBENCH_2D_ARRAY_DECL(D_outputFromGpu,DATA_TYPE,NI,NL,ni,nl);
	
	/* Initialize array(s). */
  	init_array(ni, nj, nk, nl, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));
	
	initCUDA(argc, argv);

	checkCudaErrors(cuModuleGetFunction(&(kernels[0]), cuModule, "mm2_kernel1"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[1]), cuModule, "mm2_kernel2"));

	mm2Cuda(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), 
		POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_outputFromGpu));

	/* Start timer. */
	polybench_start_instruments;

	mm2_cpu(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

	printf("CPU Time in seconds:\n");
	polybench_stop_instruments;
	polybench_print_instruments;

	compareResults(ni, nl, POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_outputFromGpu));

	POLYBENCH_FREE_ARRAY(tmp);
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(C);
	POLYBENCH_FREE_ARRAY(D);
	POLYBENCH_FREE_ARRAY(D_outputFromGpu);

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