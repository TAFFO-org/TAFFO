/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include <cuda.h>
#include <builtin_types.h>
#include <helper_cuda_drvapi.h>
#include <helper_image.h>
#include <helper_string.h>
#include <helper_timer.h>

#define POLYBENCH_TIME 1

#include "fdtd2d.cuh"
#include "fdtd2d_sh_ann.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 0

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
	#define PTX_FILE "fdtd2d.ptx"
#else
	#define PTX_FILE "fdtd2d.taffo.ptx"
#endif
#endif

#define RUN_ON_CPU

static int initCUDA(int argc, char **argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernels[3];
size_t totalGlobalMem;

const char *sSDKsample = "PolyBench fdtd2d (Driver API)";

void init_arrays(int tmax, int nx, int ny, DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX), DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny), 
		DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
 	int i;
  	int j;
  	for (i = 0; i < tmax; i++)
	{
		DATA_TYPE tmp = i;
		_fict_[i] = tmp;
	}
	
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			int a = i*(j+1) + 1;
			int b = (i-1)*(j+2) + 2;
			int c = (i-9)*(j+4) + 3;
			DATA_TYPE d = a / (NX * NY);
			DATA_TYPE e = b / (NX * NY);
			DATA_TYPE f = c / (NX * NY);
			ex[i][j] = d;
			ey[i][j] = e;
			hz[i][j] = f;
		}
	}
}


void runFdtd(int tmax, int nx, int ny, DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX), DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny), 
	DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
	int t, i, j;
	
	for (t=0; t < _PB_TMAX; t++)  
	{
		for (j=0; j < _PB_NY; j++)
		{
			ey[0][j] = _fict_[t];
		}
	
		for (i = 1; i < _PB_NX; i++)
		{
       		for (j = 0; j < _PB_NY; j++)
			{
       			ey[i][j] = ey[i][j] - 0.5*(hz[i][j] - hz[(i-1)][j]);
        		}
		}

		for (i = 0; i < _PB_NX; i++)
		{
       		for (j = 1; j < _PB_NY; j++)
			{
				ex[i][j] = ex[i][j] - 0.5*(hz[i][j] - hz[i][(j-1)]);
			}
		}

		for (i = 0; i < _PB_NX-1; i++)
		{
			for (j = 0; j < _PB_NY-1; j++)
			{
				hz[i][j] = hz[i][j] - 0.7*(ex[i][(j+1)] - ex[i][j] + ey[(i+1)][j] - ey[i][j]);
			}
		}
	}
}


void compareResults(int nx, int ny, DATA_TYPE POLYBENCH_2D(hz1,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz2,NX,NY,nx,ny))
{
	int i, j, fail;
	fail = 0;
	
	for (i=0; i < nx; i++) 
	{
		for (j=0; j < ny; j++) 
		{
			if (percentDiff(hz1[i][j], hz2[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void fdtdCuda(int tmax, int nx, int ny, DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX), DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny), 
	DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz_outputFromGpu,NX,NY,nx,ny))
{
	CUdeviceptr _fict_gpu;
	CUdeviceptr ex_gpu;
	CUdeviceptr ey_gpu;
	CUdeviceptr hz_gpu;

	checkCudaErrors(cuMemAlloc(&_fict_gpu, sizeof(DATA_TYPE) * TMAX));
	checkCudaErrors(cuMemAlloc(&ex_gpu, sizeof(DATA_TYPE) * NX * NY));	
  	checkCudaErrors(cuMemAlloc(&ey_gpu, sizeof(DATA_TYPE) * NX * NY));
	checkCudaErrors(cuMemAlloc(&hz_gpu, sizeof(DATA_TYPE) * NX * NY));	

	checkCudaErrors(cuMemcpyHtoD(_fict_gpu, _fict_, sizeof(DATA_TYPE) * TMAX));
	checkCudaErrors(cuMemcpyHtoD(ex_gpu, ex, sizeof(DATA_TYPE) * NX * NY));	
	checkCudaErrors(cuMemcpyHtoD(ey_gpu, ey, sizeof(DATA_TYPE) * NX * NY));
	checkCudaErrors(cuMemcpyHtoD(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY));	

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid( (size_t)ceil(((float)NY) / ((float)block.x)), (size_t)ceil(((float)NX) / ((float)block.y)));

	/* Start timer. */
  	polybench_start_instruments;

	for(int t = 0; t < _PB_TMAX; t++)
	{
		void *args1[7] = {&nx, &ny, &_fict_gpu, &ex_gpu, &ey_gpu, &hz_gpu, &t};
		checkCudaErrors(cuLaunchKernel(
        kernels[0], grid.x, grid.y, grid.z, block.x, block.y, block.z,
        0, NULL, args1, NULL));
		checkCudaErrors(cuCtxSynchronize());

		void *args2[6] = {&nx, &ny, &ex_gpu, &ey_gpu, &hz_gpu, &t};
		checkCudaErrors(cuLaunchKernel(
        kernels[1], grid.x, grid.y, grid.z, block.x, block.y, block.z,
        0, NULL, args2, NULL));
		checkCudaErrors(cuCtxSynchronize());

		void *args3[6] = {&nx, &ny, &ex_gpu, &ey_gpu, &hz_gpu, &t};
		checkCudaErrors(cuLaunchKernel(
        kernels[2], grid.x, grid.y, grid.z, block.x, block.y, block.z,
        0, NULL, args3, NULL));
		checkCudaErrors(cuCtxSynchronize());
	}
	
	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	checkCudaErrors(cuMemcpyDtoH(hz_outputFromGpu, hz_gpu, sizeof(DATA_TYPE) * NX * NY));	
		
	checkCudaErrors(cuMemFree(_fict_gpu));
	checkCudaErrors(cuMemFree(ex_gpu));
	checkCudaErrors(cuMemFree(ey_gpu));
	checkCudaErrors(cuMemFree(hz_gpu));
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
         fprintf(stderr, DATA_PRINTF_MODIFIER, hz[i][j]);
      if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


int main(int argc, char *argv[])
{
	int tmax = TMAX;
	int nx = NX;
	int ny = NY;

	ANN_FICT POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,TMAX);
	ANN_EX POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny);
	ANN_EY POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE,NX,NY,nx,ny);
	ANN_HZ POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE,NX,NY,nx,ny);
	ANN_HZ POLYBENCH_2D_ARRAY_DECL(hz_outputFromGpu,DATA_TYPE,NX,NY,nx,ny);

	init_arrays(tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz));

	initCUDA(argc, argv);
	checkCudaErrors(cuModuleGetFunction(&(kernels[0]), cuModule, "fdtd_step1_kernel"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[1]), cuModule, "fdtd_step2_kernel"));
	checkCudaErrors(cuModuleGetFunction(&(kernels[2]), cuModule, "fdtd_step3_kernel"));

	fdtdCuda(tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(hz_outputFromGpu));


	/* Start timer. */
	polybench_start_instruments;

	//runFdtd(tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz));

	/* Stop and print timer. */
	printf("CPU Time in seconds:\n");
	polybench_stop_instruments;
	polybench_print_instruments;
	
	//compareResults(nx, ny, POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(hz_outputFromGpu));

	print_array(nx, ny, POLYBENCH_ARRAY(hz_outputFromGpu));

	POLYBENCH_FREE_ARRAY(_fict_);
	POLYBENCH_FREE_ARRAY(ex);
	POLYBENCH_FREE_ARRAY(ey);
	POLYBENCH_FREE_ARRAY(hz);
	POLYBENCH_FREE_ARRAY(hz_outputFromGpu);

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
  //printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

  checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, cuDevice));
  //printf("  Total amount of global memory:     %llu bytes\n",
  //       (long long unsigned int)totalGlobalMem);

  checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

  // Create module from binary file (PTX)
  checkCudaErrors(cuModuleLoad(&cuModule, PTX_FILE));

  return 0;
}

#include <polybench.c>