/**
 * mvt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <assert.h>
#include <builtin_types.h>
#include <cuda.h>
#include <helper_cuda_drvapi.h>
#include <helper_image.h>
#include <helper_string.h>
#include <helper_timer.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define POLYBENCH_TIME 1

#include "mvt.cuh"
#include "mvt_sh_ann.h"

#include <polybench.h>
#include <polybenchUtilFuncts.h>

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
#define PTX_FILE "mvt.ptx"
#else
#define PTX_FILE "mvt.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char** argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernels[2];
size_t totalGlobalMem;

const char* sSDKsample = "PolyBench mvt (Driver API)";

void init_array(int n,
                DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                DATA_TYPE POLYBENCH_1D(x1, N, n),
                DATA_TYPE POLYBENCH_1D(x2, N, n),
                DATA_TYPE POLYBENCH_1D(y1, N, n),
                DATA_TYPE POLYBENCH_1D(y2, N, n)) {
  int i, j;
  DATA_TYPE tmp;

  for (i = 0; i < n; i++) {
    tmp = ((DATA_TYPE) i) / N;
    x1[i] = tmp;
    tmp = ((DATA_TYPE) i + 1) / N;
    x2[i] = tmp;
    tmp = ((DATA_TYPE) i + 3) / N;
    y1[i] = tmp;
    tmp = ((DATA_TYPE) i + 4) / N;
    y2[i] = tmp;
    for (j = 0; j < n; j++) {
      tmp = ((DATA_TYPE) i * j) / (N * N);
      A[i][j] = tmp;
    }
  }
}

void runMvt(int n,
            DATA_TYPE POLYBENCH_2D(a, N, N, n, n),
            DATA_TYPE POLYBENCH_1D(x1, N, n),
            DATA_TYPE POLYBENCH_1D(x2, N, n),
            DATA_TYPE POLYBENCH_1D(y1, N, n),
            DATA_TYPE POLYBENCH_1D(y2, N, n)) {
  int i, j;

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < N; j++)
      x1[i] = x1[i] + a[i][j] * y1[j];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x2[i] = x2[i] + a[j][i] * y2[j];
}

void compareResults(int n,
                    DATA_TYPE POLYBENCH_1D(x1, N, n),
                    DATA_TYPE POLYBENCH_1D(x1_outputFromGpu, N, n),
                    DATA_TYPE POLYBENCH_1D(x2, N, n),
                    DATA_TYPE POLYBENCH_1D(x2_outputFromGpu, N, n)) {
  int i, fail;
  fail = 0;

  for (i = 0; i < n; i++) {
    if (percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
      fail++;

    if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
      fail++;
  }

  // Print results
  printf(
    "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void mvtCuda(int n,
             DATA_TYPE POLYBENCH_2D(a, N, N, n, n),
             DATA_TYPE POLYBENCH_1D(x1, N, n),
             DATA_TYPE POLYBENCH_1D(x2, N, n),
             DATA_TYPE POLYBENCH_1D(y_1, N, n),
             DATA_TYPE POLYBENCH_1D(y_2, N, n),
             DATA_TYPE POLYBENCH_1D(x1_outputFromGpu, N, n),
             DATA_TYPE POLYBENCH_1D(x2_outputFromGpu, N, n)) {
  CUdeviceptr a_gpu;
  CUdeviceptr x1_gpu;
  CUdeviceptr x2_gpu;
  CUdeviceptr y_1_gpu;
  CUdeviceptr y_2_gpu;

  checkCudaErrors(cuMemAlloc(&a_gpu, sizeof(DATA_TYPE) * N * N));
  checkCudaErrors(cuMemAlloc(&x1_gpu, sizeof(DATA_TYPE) * N));
  checkCudaErrors(cuMemAlloc(&x2_gpu, sizeof(DATA_TYPE) * N));
  checkCudaErrors(cuMemAlloc(&y_1_gpu, sizeof(DATA_TYPE) * N));
  checkCudaErrors(cuMemAlloc(&y_2_gpu, sizeof(DATA_TYPE) * N));

  checkCudaErrors(cuMemcpyHtoD(a_gpu, a, sizeof(DATA_TYPE) * N * N));
  checkCudaErrors(cuMemcpyHtoD(x1_gpu, x1, sizeof(DATA_TYPE) * N));
  checkCudaErrors(cuMemcpyHtoD(x2_gpu, x2, sizeof(DATA_TYPE) * N));
  checkCudaErrors(cuMemcpyHtoD(y_1_gpu, y_1, sizeof(DATA_TYPE) * N));
  checkCudaErrors(cuMemcpyHtoD(y_2_gpu, y_2, sizeof(DATA_TYPE) * N));

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t) ceil((float) N / ((float) DIM_THREAD_BLOCK_X)), 1);

  /* Start timer. */
  polybench_start_instruments;

  void* args1[4] = {&n, &a_gpu, &x1_gpu, &y_1_gpu};
  checkCudaErrors(cuLaunchKernel(kernels[0], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args1, NULL));
  checkCudaErrors(cuCtxSynchronize());

  void* args2[4] = {&n, &a_gpu, &x2_gpu, &y_2_gpu};
  checkCudaErrors(cuLaunchKernel(kernels[1], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args2, NULL));
  checkCudaErrors(cuCtxSynchronize());

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;

  checkCudaErrors(cuMemcpyDtoH(x1_outputFromGpu, x1_gpu, sizeof(DATA_TYPE) * N));
  checkCudaErrors(cuMemcpyDtoH(x2_outputFromGpu, x2_gpu, sizeof(DATA_TYPE) * N));

  checkCudaErrors(cuMemFree(a_gpu));
  checkCudaErrors(cuMemFree(x1_gpu));
  checkCudaErrors(cuMemFree(x2_gpu));
  checkCudaErrors(cuMemFree(y_1_gpu));
  checkCudaErrors(cuMemFree(y_2_gpu));
  checkCudaErrors(cuCtxDestroy(cuContext));
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n))

{
  int i;

  for (i = 0; i < n; i++) {
    fprintf(stderr, DATA_PRINTF_MODIFIER, x1[i]);
    fprintf(stderr, DATA_PRINTF_MODIFIER, x2[i]);
    if (i % 20 == 0)
      fprintf(stderr, "\n");
  }
}

int main(int argc, char* argv[]) {
  int n = N;

  ANN_A POLYBENCH_2D_ARRAY_DECL(a, DATA_TYPE, N, N, n, n);
  ANN_X1 POLYBENCH_1D_ARRAY_DECL(x1, DATA_TYPE, N, n);
  ANN_X2 POLYBENCH_1D_ARRAY_DECL(x2, DATA_TYPE, N, n);
  ANN_X1 POLYBENCH_1D_ARRAY_DECL(x1_outputFromGpu, DATA_TYPE, N, n);
  ANN_X2 POLYBENCH_1D_ARRAY_DECL(x2_outputFromGpu, DATA_TYPE, N, n);
  ANN_Y_1 POLYBENCH_1D_ARRAY_DECL(y_1, DATA_TYPE, N, n);
  ANN_Y_2 POLYBENCH_1D_ARRAY_DECL(y_2, DATA_TYPE, N, n);

  init_array(
    n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2));

  initCUDA(argc, argv);

  checkCudaErrors(cuModuleGetFunction(&(kernels[0]), cuModule, "mvt_kernel1"));
  checkCudaErrors(cuModuleGetFunction(&(kernels[1]), cuModule, "mvt_kernel2"));

  mvtCuda(n,
          POLYBENCH_ARRAY(a),
          POLYBENCH_ARRAY(x1),
          POLYBENCH_ARRAY(x2),
          POLYBENCH_ARRAY(y_1),
          POLYBENCH_ARRAY(y_2),
          POLYBENCH_ARRAY(x1_outputFromGpu),
          POLYBENCH_ARRAY(x2_outputFromGpu));

  /* Start timer. */
  polybench_start_instruments;

  // run the algorithm on the CPU
  // runMvt(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1),
  // POLYBENCH_ARRAY(y_2));

  /* Stop and print timer. */
  printf("CPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;
  // compareResults(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x1_outputFromGpu), POLYBENCH_ARRAY(x2),
  // POLYBENCH_ARRAY(x2_outputFromGpu));

  print_array(n, POLYBENCH_ARRAY(x1_outputFromGpu), POLYBENCH_ARRAY(x2_outputFromGpu));

  POLYBENCH_FREE_ARRAY(a);
  POLYBENCH_FREE_ARRAY(x1);
  POLYBENCH_FREE_ARRAY(x2);
  POLYBENCH_FREE_ARRAY(x1_outputFromGpu);
  POLYBENCH_FREE_ARRAY(x2_outputFromGpu);
  POLYBENCH_FREE_ARRAY(y_1);
  POLYBENCH_FREE_ARRAY(y_2);

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
