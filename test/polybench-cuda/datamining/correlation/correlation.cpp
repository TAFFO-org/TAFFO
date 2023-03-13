/**
 * correlation.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "correlation.cuh"
#include "correlation_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 15.0

#define GPU_DEVICE 0

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#define FLOAT_N 3214212.01f
#define EPS 0.005f

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
	#define PTX_FILE "correlation.ptx"
#else
	#define PTX_FILE "correlation.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char **argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernels[4];
size_t totalGlobalMem;

const char *sSDKsample = "PolyBench correlation (Driver API)";

double frand(void)
{
	return (double)rand() / (double)RAND_MAX;
}

void init_arrays(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n), DATA_TYPE* float_n, DATA_TYPE* eps, DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n))
{
    int i, j;
	*float_n = 3214212.01;
	*eps = 0.005;

	for (i=0; i < m; i++) 
	{
    	for (j=0; j < n; j++) 
		{
        	DATA_TYPE d = (DATA_TYPE)frand();
       		data[i][j] = (DATA_TYPE)d;
			symmat[i][j] = 0;
			//fprintf(stderr, "%f\n", data[i][j]);
       	}
    	}
}


void correlation(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n), DATA_TYPE POLYBENCH_1D(mean, M, m), DATA_TYPE POLYBENCH_1D(stddev, M, m),
		DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n))
{
	int i, j, j1, j2;	
	
	// Determine mean of column vectors of input data matrix 
  	for (j = 0; j < _PB_M; j++)
   	{
  		mean[j] = 0.0;

   		for (i = 0; i < _PB_N; i++)
		{
			mean[j] += data[i][j];
   		}
		//fprintf(stderr, "%lf\t",mean[j]);
		mean[j] /= (DATA_TYPE)FLOAT_N;
		//fprintf(stderr, "%lf\n",mean[j]);
   	}

	// Determine standard deviations of column vectors of data matrix. 
  	for (j = 0; j < _PB_M; j++)
   	{
   		stddev[j] = 0.0;
      
		for (i = 0; i < _PB_N; i++)
		{
			stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
		}
		
		stddev[j] /= FLOAT_N;
		stddev[j] = sqrt_of_array_cell(stddev, j);
		stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
	}

 	// Center and reduce the column vectors. 
  	for (i = 0; i < _PB_N; i++)
	{
		for (j = 0; j < _PB_M; j++)
		{
			data[i][j] -= mean[j];
			__attribute__((annotate("scalar(range(-100, 100) final)"))) DATA_TYPE tmp = sqrt(FLOAT_N)*stddev[j];
			data[i][j] /= tmp;
		}
	}

	// Calculate the m * m correlation matrix. 
  	for (j1 = 0; j1 < _PB_M-1; j1++)
	{	
		symmat[j1][j1] = 1.0;
    
		for (j2 = j1+1; j2 < _PB_M; j2++)
		{
	  		symmat[j1][j2] = 0.0;

	  		for (i = 0; i < _PB_N; i++)
			{
	   			symmat[j1][j2] += (data[i][j1] * data[i][j2]);
			}

	  		symmat[j2][j1] = symmat[j1][j2];
		}
	}
 
	symmat[M-1][M-1] = 1.0;
}


void compareResults(int m, int n, DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n), DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu, M, N, m, n))
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < m; i++)
	{
		for (j=0; j < n; j++)
		{
			//fprintf(stderr, "%d, %d, %lf, %lf\n", i, j, symmat[i][j], symmat_outputFromGpu[i][j]);
			DATA_TYPE a = symmat[i][j];
			a = 1.00f;
			DATA_TYPE b = symmat_outputFromGpu[i][j];
			//fprintf(stderr, "%d, %d, %lf, %lf, %lf\n", i, j, a, b, percentDiff(a, b));
			if (percentDiff(a, b) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;		
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m))

{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array2(int m,
		 DATA_TYPE POLYBENCH_1D(symmat,M,m))

{
  int i, j;


    for (j = 0; j < m; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[j]);
      if (( j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}

void correlationCuda(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n), DATA_TYPE POLYBENCH_1D(mean, M, m), 
			DATA_TYPE POLYBENCH_1D(stddev, M, m), DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n), 
			DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu, M, N, m, n), DATA_TYPE float_n, DATA_TYPE eps)
{
	CUdeviceptr data_gpu;
	CUdeviceptr stddev_gpu;
	CUdeviceptr mean_gpu;
	CUdeviceptr symmat_gpu;

	ANN_FLOAT_N DATA_TYPE lfloat_n[1] = {float_n};
	ANN_EPS DATA_TYPE leps[1] = {eps};

  	checkCudaErrors(cuMemAlloc(&data_gpu, sizeof(DATA_TYPE) * M * N));
	checkCudaErrors(cuMemAlloc(&symmat_gpu, sizeof(DATA_TYPE) * M * N));
	checkCudaErrors(cuMemAlloc(&stddev_gpu, sizeof(DATA_TYPE) * M));
	checkCudaErrors(cuMemAlloc(&mean_gpu, sizeof(DATA_TYPE) * M ));
  	
	checkCudaErrors(cuMemcpyHtoD(data_gpu, data, sizeof(DATA_TYPE) * M * N));
	checkCudaErrors(cuMemcpyHtoD(symmat_gpu, symmat, sizeof(DATA_TYPE) * M * N));
	checkCudaErrors(cuMemcpyHtoD(stddev_gpu, stddev, sizeof(DATA_TYPE) * M ));
	checkCudaErrors(cuMemcpyHtoD(mean_gpu, mean, sizeof(DATA_TYPE) * M ));
		
	dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
	dim3 grid1((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)), 1);
	
	dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
	dim3 grid2((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)), 1);
	
	dim3 block3(DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y);
	dim3 grid3((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)), (size_t)(ceil((float)(N)) / ((float)DIM_THREAD_BLOCK_KERNEL_3_Y)));
	
	dim3 block4(DIM_THREAD_BLOCK_KERNEL_4_X, DIM_THREAD_BLOCK_KERNEL_4_Y);
	dim3 grid4((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_4_X)), 1);

	/* Start timer. */
  	polybench_start_instruments;

	void *args1[5] = {&m, &n, &mean_gpu, &data_gpu, &lfloat_n};
	checkCudaErrors(cuLaunchKernel(
        kernels[0], grid1.x, grid1.y, grid1.z, block1.x, block1.y, block1.z,
        0, NULL, args1, NULL));
	checkCudaErrors(cuCtxSynchronize());

	void *args2[7] = {&m, &n, &mean_gpu, &stddev_gpu, &data_gpu, &lfloat_n, &leps};
	checkCudaErrors(cuLaunchKernel(
        kernels[1], grid2.x, grid2.y, grid2.z, block2.x, block2.y, block2.z,
        0, NULL, args2, NULL));
	checkCudaErrors(cuCtxSynchronize());

	/* checkCudaErrors(cuMemcpyDtoH(stddev, stddev_gpu, sizeof(DATA_TYPE) * M ));
	print_array2(m, stddev); */

	void *args3[6] = {&m, &n, &mean_gpu, &stddev_gpu, &data_gpu, &lfloat_n};
	checkCudaErrors(cuLaunchKernel(
        kernels[2], grid3.x, grid3.y, grid3.z, block3.x, block3.y, block3.z,
        0, NULL, args3, NULL));
	checkCudaErrors(cuCtxSynchronize());

	//checkCudaErrors(cuMemcpyDtoH(data, data_gpu, sizeof(DATA_TYPE) * M * N));
	//print_array(m, data);

	void *args4[4] = {&m, &n, &symmat_gpu, &data_gpu};
	checkCudaErrors(cuLaunchKernel(
        kernels[3], grid4.x, grid4.y, grid4.z, block4.x, block4.y, block4.z,
        0, NULL, args4, NULL));
	checkCudaErrors(cuCtxSynchronize());

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	/*DATA_TYPE valueAtSymmatIndexMTimesMPlus1PlusMPoint = 1.0;
	cudaMemcpy(&(symmat_gpu[(M-1)*M + (M-1)]), &valueAtSymmatIndexMTimesMPlus1PlusMPoint, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);*/

	checkCudaErrors(cuMemcpyDtoH(symmat_outputFromGpu, symmat_gpu, sizeof(DATA_TYPE) * M * N));
	
	checkCudaErrors(cuMemFree(data_gpu));
  	checkCudaErrors(cuMemFree(symmat_gpu));
  	checkCudaErrors(cuMemFree(stddev_gpu));
	checkCudaErrors(cuMemFree(mean_gpu));
  	checkCudaErrors(cuCtxDestroy(cuContext));
}


int main(int argc, char** argv)
{
	int m = M;
	int n = N;

	DATA_TYPE float_n;
	DATA_TYPE eps;

	POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE ANN_DATA,M,N,m,n);
  	POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE ANN_MEAN,M,m);
  	POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE ANN_STD,M,m);
	POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE ANN_SYMMAT,M,N,m,n);
  	POLYBENCH_2D_ARRAY_DECL(symmat_outputFromGpu,DATA_TYPE ANN_SYMMAT,M,N,m,n);
  	
	init_arrays(m, n, POLYBENCH_ARRAY(data), &float_n, &eps, POLYBENCH_ARRAY(symmat));
    
	initCUDA(argc, argv);

	checkCudaErrors(cuModuleGetFunction(&(kernels[0]), cuModule, "mean_kernel"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[1]), cuModule, "std_kernel"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[2]), cuModule, "reduce_kernel"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[3]), cuModule, "corr_kernel"));

	correlationCuda(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev), POLYBENCH_ARRAY(symmat), 
		POLYBENCH_ARRAY(symmat_outputFromGpu), float_n, eps);


	/* Start timer. */
	polybench_start_instruments;

	//correlation(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev), POLYBENCH_ARRAY(symmat));

	/* Stop and print timer. */
	printf("CPU Time in seconds:\n");
	polybench_stop_instruments;
	polybench_print_instruments;

	//compareResults(m, n, POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(symmat_outputFromGpu));

	print_array(m, POLYBENCH_ARRAY(symmat_outputFromGpu));	

	POLYBENCH_FREE_ARRAY(data);
	POLYBENCH_FREE_ARRAY(mean);
	POLYBENCH_FREE_ARRAY(stddev);
	POLYBENCH_FREE_ARRAY(symmat);
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