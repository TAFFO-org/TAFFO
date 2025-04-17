/**
 * jacobi1D.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define POLYBENCH_TIME 1

#include "jacobi1D.cuh"
#include "jacobi1D_sh_ann.h"

#include <polybench.h>
#include <polybenchUtilFuncts.h>

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
#define PTX_FILE "jacobi1D.ptx"
#else
#define PTX_FILE "jacobi1D.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char** argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernels[2];
size_t totalGlobalMem;

const char* sSDKsample = "PolyBench jacobi1d (Driver API)";

void init_array(int n, DATA_TYPE POLYBENCH_1D(A, N, n), DATA_TYPE POLYBENCH_1D(B, N, n)) {
  int i;

  for (i = 0; i < n; i++) {
    DATA_TYPE a = ((DATA_TYPE) 4 * i + 10) / N;
    DATA_TYPE b = ((DATA_TYPE) 7 * i + 11) / N;
    A[i] = a;
    B[i] = b;
  }
}

void runJacobi1DCpu(int tsteps, int n, DATA_TYPE POLYBENCH_1D(A, N, n), DATA_TYPE POLYBENCH_1D(B, N, n)) {
  for (int t = 0; t < _PB_TSTEPS; t++) {
    for (int i = 2; i < _PB_N - 1; i++)
      B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);

    for (int j = 2; j < _PB_N - 1; j++)
      A[j] = B[j];
  }
}

void compareResults(int n,
                    DATA_TYPE POLYBENCH_1D(a, N, n),
                    DATA_TYPE POLYBENCH_1D(a_outputFromGpu, N, n),
                    DATA_TYPE POLYBENCH_1D(b, N, n),
                    DATA_TYPE POLYBENCH_1D(b_outputFromGpu, N, n)) {
  int i, fail;
  fail = 0;

  // Compare a and c
  for (i = 0; i < n; i++)
    if (percentDiff(a[i], a_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
      fail++;

  for (i = 0; i < n; i++)
    if (percentDiff(b[i], b_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
      fail++;

  // Print results
  printf(
    "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void runJacobi1DCUDA(int tsteps,
                     int n,
                     DATA_TYPE POLYBENCH_1D(A, N, n),
                     DATA_TYPE POLYBENCH_1D(B, N, n),
                     DATA_TYPE POLYBENCH_1D(A_outputFromGpu, N, n),
                     DATA_TYPE POLYBENCH_1D(B_outputFromGpu, N, n)) {
  CUdeviceptr A_gpu;
  CUdeviceptr B_gpu;

  checkCudaErrors(cuMemAlloc(&A_gpu, N * sizeof(DATA_TYPE)));
  checkCudaErrors(cuMemAlloc(&B_gpu, N * sizeof(DATA_TYPE)));

  checkCudaErrors(cuMemcpyHtoD(A_gpu, A, N * sizeof(DATA_TYPE)));
  checkCudaErrors(cuMemcpyHtoD(B_gpu, B, N * sizeof(DATA_TYPE)));

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((unsigned int) ceil(((float) N) / ((float) block.x)), 1);

  /* Start timer. */
  polybench_start_instruments;

  for (int t = 0; t < _PB_TSTEPS; t++) {
    void* args1[3] = {&n, &A_gpu, &B_gpu};
    checkCudaErrors(
      cuLaunchKernel(kernels[0], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args1, NULL));
    checkCudaErrors(cuCtxSynchronize());

    void* args2[3] = {&n, &A_gpu, &B_gpu};
    checkCudaErrors(
      cuLaunchKernel(kernels[1], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args2, NULL));
    checkCudaErrors(cuCtxSynchronize());
  }

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;

  checkCudaErrors(cuMemcpyDtoH(A_outputFromGpu, A_gpu, sizeof(DATA_TYPE) * N));
  checkCudaErrors(cuMemcpyDtoH(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * N));

  checkCudaErrors(cuMemFree(A_gpu));
  checkCudaErrors(cuMemFree(B_gpu));
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, DATA_TYPE POLYBENCH_1D(A, N, n))

{
  int i;

  for (i = 0; i < n; i++) {
    fprintf(stderr, DATA_PRINTF_MODIFIER, A[i]);
    if (i % 20 == 0)
      fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

int main(int argc, char** argv) {
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  ANN_A POLYBENCH_1D_ARRAY_DECL(a, DATA_TYPE, N, n);
  ANN_B POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
  ANN_A POLYBENCH_1D_ARRAY_DECL(a_outputFromGpu, DATA_TYPE, N, n);
  ANN_B POLYBENCH_1D_ARRAY_DECL(b_outputFromGpu, DATA_TYPE, N, n);

  init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));

  initCUDA(argc, argv);
  checkCudaErrors(cuModuleGetFunction(&(kernels[0]), cuModule, "runJacobiCUDA_kernel1"));
  checkCudaErrors(cuModuleGetFunction(&(kernels[1]), cuModule, "runJacobiCUDA_kernel2"));

  runJacobi1DCUDA(tsteps,
                  n,
                  POLYBENCH_ARRAY(a),
                  POLYBENCH_ARRAY(b),
                  POLYBENCH_ARRAY(a_outputFromGpu),
                  POLYBENCH_ARRAY(b_outputFromGpu));

  /* Start timer. */
  polybench_start_instruments;

  // runJacobi1DCpu(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));

  /* Stop and print timer. */
  printf("CPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;

  // compareResults(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b),
  // POLYBENCH_ARRAY(b_outputFromGpu));

  print_array(n, POLYBENCH_ARRAY(a_outputFromGpu));

  POLYBENCH_FREE_ARRAY(a);
  POLYBENCH_FREE_ARRAY(a_outputFromGpu);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(b_outputFromGpu);

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
  // printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

  checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, cuDevice));
  // printf("  Total amount of global memory:     %llu bytes\n",
  //        (long long unsigned int)totalGlobalMem);

  checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

  // Create module from binary file (PTX)
  checkCudaErrors(cuModuleLoad(&cuModule, PTX_FILE));

  return 0;
}

#include <polybench.c>
