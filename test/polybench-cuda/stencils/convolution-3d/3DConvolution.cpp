/**
 * 3DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "3DConvolution.cuh"
#include "3DConvolution_sh_ann.h"

#include <polybench.h>
#include <polybenchUtilFuncts.h>

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
#define PTX_FILE "3DConvolution.ptx"
#else
#define PTX_FILE "3DConvolution.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char** argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernel;
size_t totalGlobalMem;

const char* sSDKsample = "PolyBench 3DConvolution (Driver API)";

void conv3D(int ni,
            int nj,
            int nk,
            DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk),
            DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk)) {
  int i, j, k;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +2;
  c21 = +5;
  c31 = -8;
  c12 = -3;
  c22 = +6;
  c32 = -9;
  c13 = +4;
  c23 = +7;
  c33 = +10;

  for (i = 1; i < _PB_NI - 1; ++i)     // 0
  {
    for (j = 1; j < _PB_NJ - 1; ++j)   // 1
    {
      for (k = 1; k < _PB_NK - 1; ++k) // 2
      {
        B[i][j][k] =
          c11 * A[(i - 1)][(j - 1)][(k - 1)] + c13 * A[(i + 1)][(j - 1)][(k - 1)] + c21 * A[(i - 1)][(j - 1)][(k - 1)]
          + c23 * A[(i + 1)][(j - 1)][(k - 1)] + c31 * A[(i - 1)][(j - 1)][(k - 1)] + c33 * A[(i + 1)][(j - 1)][(k - 1)]
          + c12 * A[(i + 0)][(j - 1)][(k + 0)] + c22 * A[(i + 0)][(j + 0)][(k + 0)] + c32 * A[(i + 0)][(j + 1)][(k + 0)]
          + c11 * A[(i - 1)][(j - 1)][(k + 1)] + c13 * A[(i + 1)][(j - 1)][(k + 1)] + c21 * A[(i - 1)][(j + 0)][(k + 1)]
          + c23 * A[(i + 1)][(j + 0)][(k + 1)] + c31 * A[(i - 1)][(j + 1)][(k + 1)]
          + c33 * A[(i + 1)][(j + 1)][(k + 1)];
      }
    }
  }
}

double frand(void) { return (double) rand() / (double) RAND_MAX; }

void init(int ni,
          int nj,
          int nk,
          DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk),
          DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk)) {
  int i, j, k;

  for (i = 0; i < ni; ++i) {
    for (j = 0; j < nj; ++j) {
      for (k = 0; k < nk; ++k) {
        float tmp = frand();
        A[i][j][k] = tmp;
        B[i][j][k] = 0;
      }
    }
  }
}

void compareResults(int ni,
                    int nj,
                    int nk,
                    DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk),
                    DATA_TYPE POLYBENCH_3D(B_outputFromGpu, NI, NJ, NK, ni, nj, nk)) {
  int i, j, k, fail;
  fail = 0;

  // Compare result from cpu and gpu
  for (i = 1; i < ni - 1; ++i)     // 0
  {
    for (j = 1; j < nj - 1; ++j)   // 1
    {
      for (k = 1; k < nk - 1; ++k) // 2
        if (percentDiff(B[i][j][k], B_outputFromGpu[i][j][k]) > PERCENT_DIFF_ERROR_THRESHOLD)
          fail++;
    }
  }

  // Print results
  printf(
    "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void convolution3DCuda(int ni,
                       int nj,
                       int nk,
                       DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk),
                       DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk),
                       DATA_TYPE POLYBENCH_3D(B_outputFromGpu, NI, NJ, NK, ni, nj, nk)) {
  CUdeviceptr A_gpu;
  CUdeviceptr B_gpu;

  checkCudaErrors(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NI * NK * NJ));
  checkCudaErrors(cuMemAlloc(&B_gpu, sizeof(DATA_TYPE) * NK * NJ * NI));

  checkCudaErrors(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NI * NK * NJ));
  checkCudaErrors(cuMemcpyHtoD(B_gpu, B, sizeof(DATA_TYPE) * NJ * NK * NI));

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t) (ceil(((float) NK) / ((float) block.x))), (size_t) (ceil(((float) NJ) / ((float) block.y))));

  /* Start timer. */
  polybench_start_instruments;

  int i;
  for (i = 1; i < _PB_NI - 1; ++i) // 0
  {
    void* args1[6] = {&ni, &nj, &nk, &A_gpu, &B_gpu, &i};
    checkCudaErrors(cuLaunchKernel(kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args1, NULL));
    checkCudaErrors(cuCtxSynchronize());
  }
  printf("GPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;

  checkCudaErrors(cuMemcpyDtoH(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK));

  checkCudaErrors(cuMemFree(A_gpu));
  checkCudaErrors(cuMemFree(B_gpu));
  checkCudaErrors(cuCtxDestroy(cuContext));
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk)) {
  int i, j, k;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      for (k = 0; k < nk; k++) {
        fprintf(stderr, DATA_PRINTF_MODIFIER, B[i][j][k]);
        if ((i * (nj * nk) + j * nk + k) % 20 == 0)
          fprintf(stderr, "\n");
      }
  fprintf(stderr, "\n");
}

int main(int argc, char* argv[]) {
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  ANN_A POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, NI, NJ, NK, ni, nj, nk);
  ANN_B POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, NI, NJ, NK, ni, nj, nk);
  ANN_B POLYBENCH_3D_ARRAY_DECL(B_outputFromGpu, DATA_TYPE, NI, NJ, NK, ni, nj, nk);

  init(ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  initCUDA(argc, argv);

  checkCudaErrors(cuModuleGetFunction(&(kernel), cuModule, "convolution3D_kernel"));

  convolution3DCuda(ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));

  /* Start timer. */
  polybench_start_instruments;

  conv3D(ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  printf("CPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;
  compareResults(ni, nj, nk, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));

  print_array(NI, NJ, NK, POLYBENCH_ARRAY(B_outputFromGpu));

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
