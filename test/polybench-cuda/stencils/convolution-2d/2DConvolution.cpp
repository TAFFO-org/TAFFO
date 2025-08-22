/**
 * 2DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <builtin_types.h>
#include <cuda.h>
#include <helper_cuda_drvapi.h>
#include <helper_image.h>
#include <helper_string.h>
#include <helper_timer.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define POLYBENCH_TIME 1

#include "2DConvolution.cuh"
#include "2DConvolution_sh_ann.h"

#include <polybench.h>
#include <polybenchUtilFuncts.h>

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
#define PTX_FILE "2DConvolution.ptx"
#else
#define PTX_FILE "2DConvolution.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char** argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernel;
size_t totalGlobalMem;

const char* sSDKsample = "PolyBench 2DConvolution (Driver API)";

void conv2D(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj)) {
  int i, j;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +0.2;
  c21 = +0.5;
  c31 = -0.8;
  c12 = -0.3;
  c22 = +0.6;
  c32 = -0.9;
  c13 = +0.4;
  c23 = +0.7;
  c33 = +0.10;

  for (i = 1; i < _PB_NI - 1; ++i)   // 0
  {
    for (j = 1; j < _PB_NJ - 1; ++j) // 1
    {
      B[i][j] = c11 * A[(i - 1)][(j - 1)] + c12 * A[(i + 0)][(j - 1)] + c13 * A[(i + 1)][(j - 1)]
              + c21 * A[(i - 1)][(j + 0)] + c22 * A[(i + 0)][(j + 0)] + c23 * A[(i + 1)][(j + 0)]
              + c31 * A[(i - 1)][(j + 1)] + c32 * A[(i + 0)][(j + 1)] + c33 * A[(i + 1)][(j + 1)];
    }
  }
}

void init(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj)) {
  int i, j;

  for (i = 0; i < ni; ++i) {
    for (j = 0; j < nj; ++j) {
      float tmp = (float) rand() / RAND_MAX;
      A[i][j] = tmp;
    }
  }
}

void compareResults(int ni,
                    int nj,
                    DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj),
                    DATA_TYPE POLYBENCH_2D(B_outputFromGpu, NI, NJ, ni, nj)) {
  int i, j, fail;
  fail = 0;

  // Compare outputs from CPU and GPU
  for (i = 1; i < (ni - 1); i++) {
    for (j = 1; j < (nj - 1); j++)
      if (percentDiff(B[i][j], B_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
        fail++;
  }

  // Print results
  printf(
    "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void convolution2DCuda(int ni,
                       int nj,
                       DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                       DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj),
                       DATA_TYPE POLYBENCH_2D(B_outputFromGpu, NI, NJ, ni, nj)) {
  CUdeviceptr A_gpu;
  CUdeviceptr B_gpu;

  checkCudaErrors(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NI * NJ));
  checkCudaErrors(cuMemAlloc(&B_gpu, sizeof(DATA_TYPE) * NI * NJ));

  checkCudaErrors(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ));

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t) ceil(((float) NI) / ((float) block.x)), (size_t) ceil(((float) NJ) / ((float) block.y)));

  polybench_start_instruments;

  void* args1[4] = {&ni, &nj, &A_gpu, &B_gpu};
  checkCudaErrors(cuLaunchKernel(kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args1, NULL));
  checkCudaErrors(cuCtxSynchronize());

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");

  polybench_stop_instruments;
  polybench_print_instruments;

  checkCudaErrors(cuMemcpyDtoH(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ));

  checkCudaErrors(cuMemFree(A_gpu));
  checkCudaErrors(cuMemFree(B_gpu));
  checkCudaErrors(cuCtxDestroy(cuContext));
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj, DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj)) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, B[i][j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

int main(int argc, char* argv[]) {
  /* Retrieve problem size */
  int ni = NI;
  int nj = NJ;

  ANN_A POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NJ, ni, nj);
  ANN_B POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NI, NJ, ni, nj);
  ANN_B POLYBENCH_2D_ARRAY_DECL(B_outputFromGpu, DATA_TYPE, NI, NJ, ni, nj);

  // initialize the arrays
  init(ni, nj, POLYBENCH_ARRAY(A));

  initCUDA(argc, argv);

  checkCudaErrors(cuModuleGetFunction(&(kernel), cuModule, "convolution2D_kernel"));

  convolution2DCuda(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));

  /* Start timer. */
  polybench_start_instruments;

  // conv2D(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  printf("CPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;

  // compareResults(ni, nj, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));

  print_array(ni, nj, POLYBENCH_ARRAY(B_outputFromGpu));

  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(B_outputFromGpu);

  return 0;
}

static int initCUDA(int argc, char** argv) {
  CUfunction cuFunction = 0;
  int major = 0, minor = 0;
  char deviceName[100];
  cuDevice = findCudaDeviceDRV(argc, (const char**) argv);
  // get compute capabilities and the devicename
  checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
  checkCudaErrors(cuDeviceGetName(deviceName, sizeof(deviceName), cuDevice));
  printf("> GPU Device has SM %d.%d compute capability\n", major, minor);
  checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, cuDevice));
  printf("  Total amount of global memory:     %llu bytes\n", (long long unsigned int) totalGlobalMem);
  checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));
  // Create module from binary file (PTX)
  checkCudaErrors(cuModuleLoad(&cuModule, PTX_FILE));
  return 0;
}

#include <polybench.c>
