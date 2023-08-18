/**
 * doitgen.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "doitgen.cuh"
#include "doitgen_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
	#define PTX_FILE "doitgen.ptx"
#else
	#define PTX_FILE "doitgen.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char **argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernels[2];
size_t totalGlobalMem;

const char *sSDKsample = "PolyBench doitgen (Driver API)";

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_doitgenCpu(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_3D(sum,NR,NQ,NP,nr,nq,np))
{
	int r, q, p, s;

	for (r = 0; r < _PB_NR; r++)
	{
		for (q = 0; q < _PB_NQ; q++)  
		{
			for (p = 0; p < _PB_NP; p++)  
			{
				sum[r][q][p] = 0;
				for (s = 0; s < _PB_NP; s++){
					sum[r][q][p] = sum[r][q][p] + A[r][q][s] * C4[s][p];
				}	
			}
			for (p = 0; p < _PB_NR; p++)
				A[r][q][p] = sum[r][q][p];
		}
	}
}



/* Array initialization. */
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
	int i, j, k;

	for (i = 0; i < nr; i++)
		for (j = 0; j < nq; j++)
			for (k = 0; k < np; k++) {
        float tmp = ((DATA_TYPE) i*j + k) / np;
				A[i][j][k] = tmp;
      }

	for (i = 0; i < np; i++)
		for (j = 0; j < np; j++) {
      float tmp = ((DATA_TYPE) i*j) / np;
			C4[i][j] = tmp;
    }				
}


void compareResults(int nr, int nq, int np, DATA_TYPE POLYBENCH_3D(sum,NR,NQ,NP,nr,nq,np), 
			DATA_TYPE POLYBENCH_3D(sum_outputFromGpu,NR,NQ,NP,nr,nq,np))
{
	int fail = 0;
	
	for (int r = 0; r < nr; r++)
	{
    		for (int q = 0; q < nq; q++)  
		{
      			for (int p = 0; p < np; p++)  
			{
				if (percentDiff(sum[r][q][p], sum_outputFromGpu[r][q][p]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
			}
		}
	}
	
	// Print results
	printf("Number of misses: %d\n", fail);
}

void doitgenCuda(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_3D(sum_outputFromGpu,NR,NQ,NP,nr,nq,np))
{
	CUdeviceptr A_gpu;
	CUdeviceptr C4_gpu;
	CUdeviceptr sum_gpu;

	checkCudaErrors(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * nr * nq * np));
	checkCudaErrors(cuMemAlloc(&C4_gpu, sizeof(DATA_TYPE) * np * np));
	checkCudaErrors(cuMemAlloc(&sum_gpu, sizeof(DATA_TYPE) * nr * nq * np));

	checkCudaErrors(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * nr * nq * np));
	checkCudaErrors(cuMemcpyHtoD(C4_gpu, C4, sizeof(DATA_TYPE) * np * np));
	

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)np) / ((float)block.x) ), (unsigned int)ceil( ((float)nr) / ((float)block.y) ));

	/* Start timer. */
	polybench_start_instruments;	

	for (int r = 0; r < nr; r++)
	{
		void *args1[7] = {&nr, &nq, &np, &sum_gpu, &A_gpu, &C4_gpu, &r};
		checkCudaErrors(cuLaunchKernel(
    	    kernels[0], grid.x, grid.y, grid.z, block.x, block.y, block.z,
    	    0, NULL, args1, NULL));
		checkCudaErrors(cuCtxSynchronize());

		checkCudaErrors(cuLaunchKernel(
    	    kernels[1], grid.x, grid.y, grid.z, block.x, block.y, block.z,
    	    0, NULL, args1, NULL));
		checkCudaErrors(cuCtxSynchronize());
	}

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
	polybench_print_instruments;
		
	checkCudaErrors(cuMemcpyDtoH(sum_outputFromGpu, sum_gpu, sizeof(DATA_TYPE) * nr * nq * np));

  	checkCudaErrors(cuMemFree(A_gpu));
	checkCudaErrors(cuMemFree(sum_gpu));
	checkCudaErrors(cuMemFree(C4_gpu));
  	checkCudaErrors(cuCtxDestroy(cuContext));
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
	int i, j, k, c;
	c=0;

	for (i = 0; i < nr; i++)
	{
		for (j = 0; j < nq; j++)
		{
			for (k = 0; k < np; k++) 
			{
				fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j][k]);
				 c++;
				if (c % 20 == 0) fprintf (stderr, "\n");
			}
		}
	}
	fprintf (stderr, "\n");
}
	

int main(int argc, char *argv[])
{
	/* Retrieve problem size. */
	int nr = NR;
	int nq = NQ;
	int np = NP;

	/* Variable declaration/allocation. */
	ANN_A POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NR,NQ,NP,nr,nq,np);
	ANN_SUM POLYBENCH_3D_ARRAY_DECL(sum,DATA_TYPE,NR,NQ,NP,nr,nq,np);
	ANN_SUM POLYBENCH_3D_ARRAY_DECL(sum_outputFromGpu,DATA_TYPE,NR,NQ,NP,nr,nq,np);
	ANN_C4 POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE,NP,NP,np,np);

	/* Initialize array(s). */
	init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4));

	initCUDA(argc, argv);

	checkCudaErrors(cuModuleGetFunction(&(kernels[0]), cuModule, "doitgen_kernel1"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[1]), cuModule, "doitgen_kernel2"));		  

	doitgenCuda(nr, nq, np,
		POLYBENCH_ARRAY(A),
		POLYBENCH_ARRAY(C4),
		POLYBENCH_ARRAY(sum_outputFromGpu));

	/* Start timer. */
	polybench_start_instruments;

	/* Run kernel on CPU */
	//kernel_doitgenCpu(nr, nq, np, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C4), POLYBENCH_ARRAY(sum));	
	
	/* Stop and print timer. */
	printf("CPU Time in seconds:\n");
    polybench_stop_instruments;
	polybench_print_instruments;

	//compareResults(nr, nq, np, POLYBENCH_ARRAY(sum), POLYBENCH_ARRAY(sum_outputFromGpu));

	print_array(nr, nq, np, POLYBENCH_ARRAY(sum_outputFromGpu));

	/* Garbage collection */
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(sum);
	POLYBENCH_FREE_ARRAY(sum_outputFromGpu);
	POLYBENCH_FREE_ARRAY(C4);	
    
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