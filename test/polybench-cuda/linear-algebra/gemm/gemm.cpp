/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "gemm.cuh"
#include "gemm_sh_ann.h"

#include <polybench.h>
#include <polybenchUtilFuncts.h>

#define GPU_DEVICE 0

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
#define PTX_FILE "gemm.ptx"
#else
#define PTX_FILE "gemm.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char** argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernel;
size_t totalGlobalMem;

const char* sSDKsample = "PolyBench gemm (Driver API)";

void gemm(int ni,
          int nj,
          int nk,
          DATA_TYPE alpha,
          DATA_TYPE beta,
          DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
          DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
          DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj)) {
  int i, j, k;

  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NJ; j++) {
      C[i][j] *= beta;

      for (k = 0; k < _PB_NK; ++k)
        C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }
}

void init(int ni,
          int nj,
          int nk,
          DATA_TYPE* alpha,
          DATA_TYPE* beta,
          DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
          DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
          DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj)) {
  __attribute__((annotate("scalar(range(0, 1000) final)"))) int i;
  __attribute__((annotate("scalar(range(0, 1000) final)"))) int j;

  *alpha = 32412 / 115;
  *beta = 2123 / 115;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i * j % NI) / (NI);

  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i * j % NI) / (NI);

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) (i * j % NI) / (NI);
}

void compareResults(int ni,
                    int nj,
                    DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
                    DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NJ, ni, nj)) {
  int i, j, fail;
  fail = 0;

  // Compare CPU and GPU outputs
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++)
      if (percentDiff(C[i][j], C_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
        fail++;
  }

  // Print results
  printf(
    "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void gemmCuda(int ni,
              int nj,
              int nk,
              DATA_TYPE alpha,
              DATA_TYPE beta,
              DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
              DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
              DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
              DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NJ, ni, nj)) {
  CUdeviceptr A_gpu;
  CUdeviceptr B_gpu;
  CUdeviceptr C_gpu;

  DATA_TYPE ANN_ALPHA alpha_l[1] = {alpha};
  DATA_TYPE ANN_BETA beta_l[1] = {beta};

  checkCudaErrors(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NI * NK));
  checkCudaErrors(cuMemAlloc(&B_gpu, sizeof(DATA_TYPE) * NK * NJ));
  checkCudaErrors(cuMemAlloc(&C_gpu, sizeof(DATA_TYPE) * NI * NJ));

  checkCudaErrors(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NI * NK));
  checkCudaErrors(cuMemcpyHtoD(B_gpu, B, sizeof(DATA_TYPE) * NJ * NK));
  checkCudaErrors(cuMemcpyHtoD(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ));

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t) (ceil(((float) NI) / ((float) block.x))), (size_t) (ceil(((float) NJ) / ((float) block.y))));

  /* Start timer. */
  polybench_start_instruments;

  void* args1[8] = {&ni, &nj, &nk, &alpha_l, &beta_l, &A_gpu, &B_gpu, &C_gpu};
  checkCudaErrors(cuLaunchKernel(kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args1, NULL));
  checkCudaErrors(cuCtxSynchronize());

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;

  checkCudaErrors(cuMemcpyDtoH(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ));

  checkCudaErrors(cuMemFree(A_gpu));
  checkCudaErrors(cuMemFree(B_gpu));
  checkCudaErrors(cuMemFree(C_gpu));
  checkCudaErrors(cuCtxDestroy(cuContext));
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj, DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj)) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, C[i][j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

int main(int argc, char* argv[]) {
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  ANN_A POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  ANN_B POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  ANN_C POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NI, NJ, ni, nj);
  ANN_C POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu, DATA_TYPE, NI, NJ, ni, nj);

  init(ni, nj, nk, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

  initCUDA(argc, argv);

  checkCudaErrors(cuModuleGetFunction(&(kernel), cuModule, "gemm_kernel"));

  gemmCuda(ni,
           nj,
           nk,
           alpha,
           beta,
           POLYBENCH_ARRAY(A),
           POLYBENCH_ARRAY(B),
           POLYBENCH_ARRAY(C),
           POLYBENCH_ARRAY(C_outputFromGpu));

  /* Start timer. */
  polybench_start_instruments;

  // gemm(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

  /* Stop and print timer. */
  printf("CPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;
  // compareResults(ni, nj, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

  print_array(ni, nj, POLYBENCH_ARRAY(C_outputFromGpu));

  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(C_outputFromGpu);

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
