/**
 * covariance.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "covariance.cuh"
#include "covariance_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 0

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#define FLOAT_N 3214212.01
#define EPS 0.005

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
	#define PTX_FILE "covariance.ptx"
#else
	#define PTX_FILE "covariance.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char **argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernels[3];
size_t totalGlobalMem;

const char *sSDKsample = "PolyBench correlation (Driver API)";

void init_arrays(int m, int n, DATA_TYPE POLYBENCH_2D(data,M,N,m,n))
{
	__attribute__((annotate("scalar(range(0, 4000) final)"))) int i;
	__attribute__((annotate("scalar(range(0, 4000) final)"))) int j;

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			data[i][j] = ((DATA_TYPE) i*j) / M;
		}
	}
}


void covariance(int m, int n, DATA_TYPE POLYBENCH_2D(data,M,N,m,n), DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m), DATA_TYPE POLYBENCH_1D(mean,M,m))
{
	int i, j, j1,j2;

  	/* Determine mean of column vectors of input data matrix */
	for (j = 0; j < _PB_M; j++)
	{
		mean[j] = 0.0;
		for (i = 0; i < _PB_N; i++)
		{
        		mean[j] += data[i][j];
		}
		mean[j] /= FLOAT_N;
	}

  	/* Center the column vectors. */
	for (i = 0; i < _PB_N; i++)
	{
		for (j = 0; j < _PB_M; j++)
		{
			data[i][j] -= mean[j];
		}
	}

  	/* Calculate the m * m covariance matrix. */
	for (j1 = 0; j1 < _PB_M; j1++)
	{
		for (j2 = j1; j2 < _PB_M; j2++)
     		{
       		symmat[j1][j2] = 0.0;
			for (i = 0; i < _PB_N; i++)
			{
				symmat[j1][j2] += data[i][j1] * data[i][j2];
			}
        		symmat[j2][j1] = symmat[j1][j2];
      		}
	}
}


void compareResults(int m, int n, DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m), DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu,M,M,m,m))
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < m; i++)
	{
		for (j=0; j < n; j++)
		{
			DATA_TYPE a = symmat[i][j];
			DATA_TYPE b =  symmat_outputFromGpu[i][j];
			if (percentDiff(a, b) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}			
		}
	}
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void covarianceCuda(int m, int n, DATA_TYPE POLYBENCH_2D(data,M,N,m,n), DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m), DATA_TYPE POLYBENCH_1D(mean,M,m), 
		DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu,M,M,m,m))
{
	CUdeviceptr data_gpu;
	CUdeviceptr mean_gpu;
	CUdeviceptr symmat_gpu;


  	checkCudaErrors(cuMemAlloc(&data_gpu, sizeof(DATA_TYPE) * M * N));
	checkCudaErrors(cuMemAlloc(&symmat_gpu, sizeof(DATA_TYPE) * M * N));
	checkCudaErrors(cuMemAlloc(&mean_gpu, sizeof(DATA_TYPE) * M ));

	checkCudaErrors(cuMemcpyHtoD(data_gpu, data, sizeof(DATA_TYPE) * M * N));
	checkCudaErrors(cuMemcpyHtoD(symmat_gpu, symmat, sizeof(DATA_TYPE) * M * N));
	checkCudaErrors(cuMemcpyHtoD(mean_gpu, mean, sizeof(DATA_TYPE) * M ));
	
	dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
	dim3 grid1((size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)), 1);
	
	dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
	dim3 grid2((size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)), (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)));
	
	dim3 block3(DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y);
	dim3 grid3((size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)), 1);
	
	/* Start timer. */
  	polybench_start_instruments;

	void *args1[4] = {&m, &n, &mean_gpu, &data_gpu};
	checkCudaErrors(cuLaunchKernel(
        kernels[0], grid1.x, grid1.y, grid1.z, block1.x, block1.y, block1.z,
        0, NULL, args1, NULL));
	checkCudaErrors(cuCtxSynchronize());

	void *args2[4] = {&m, &n, &mean_gpu, &data_gpu};
	checkCudaErrors(cuLaunchKernel(
        kernels[1], grid2.x, grid2.y, grid2.z, block2.x, block2.y, block2.z,
        0, NULL, args2, NULL));
	checkCudaErrors(cuCtxSynchronize());

	void *args3[4] = {&m, &n, &symmat_gpu, &data_gpu};
	checkCudaErrors(cuLaunchKernel(
        kernels[2], grid3.x, grid3.y, grid3.z, block3.x, block3.y, block3.z,
        0, NULL, args3, NULL));
	checkCudaErrors(cuCtxSynchronize());	
	
	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	checkCudaErrors(cuMemcpyDtoH(symmat_outputFromGpu, symmat_gpu, sizeof(DATA_TYPE) * M * N));
	
	checkCudaErrors(cuMemFree(data_gpu));
  	checkCudaErrors(cuMemFree(symmat_gpu));
	checkCudaErrors(cuMemFree(mean_gpu));
  	checkCudaErrors(cuCtxDestroy(cuContext));
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char** argv)
{
	int m = M;
	int n = N;

	ANN_DATA POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
	ANN_SYMMAT POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,M,m,m);
	ANN_MEAN POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
	ANN_SYMMAT POLYBENCH_2D_ARRAY_DECL(symmat_outputFromGpu,DATA_TYPE,M,M,m,m);	

	init_arrays(m, n, POLYBENCH_ARRAY(data));
    
	initCUDA(argc, argv);

	checkCudaErrors(cuModuleGetFunction(&(kernels[0]), cuModule, "mean_kernel"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[1]), cuModule, "reduce_kernel"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[2]), cuModule, "covar_kernel"));

	covarianceCuda(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(symmat_outputFromGpu));
	

	/* Start timer. */
	polybench_start_instruments;

	covariance(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(mean));

	/* Stop and print timer. */
	printf("CPU Time in seconds:\n");
	polybench_stop_instruments;
	polybench_print_instruments;

	compareResults(m, n, POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(symmat_outputFromGpu));


	POLYBENCH_FREE_ARRAY(data);
	POLYBENCH_FREE_ARRAY(symmat);
	POLYBENCH_FREE_ARRAY(mean);
	POLYBENCH_FREE_ARRAY(symmat_outputFromGpu);	

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