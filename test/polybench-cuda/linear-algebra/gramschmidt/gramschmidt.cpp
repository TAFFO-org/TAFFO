/**
 * gramschmidt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "gramschmidt.cuh"
#include "gramschmidt_sh_ann.h"

#include <polybench.h>
#include <polybenchUtilFuncts.h>

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
#define PTX_FILE "gramschmidt.ptx"
#else
#define PTX_FILE "gramschmidt.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char** argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernels[3];
size_t totalGlobalMem;

const char* sSDKsample = "PolyBench gramschmidt (Driver API)";

double frand(void) { return (double) rand() / (double) RAND_MAX; }

void gramschmidt(int ni,
                 int nj,
                 DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                 DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                 DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj)) {
  int i, j, k;
  DATA_TYPE nrm;
  for (k = 0; k < _PB_NJ; k++) {
    nrm = 0;
    for (i = 0; i < _PB_NI; i++)
      nrm += A[i][k] * A[i][k];

    R[k][k] = sqrt(nrm);
    for (i = 0; i < _PB_NI; i++)
      Q[i][k] = A[i][k] / R[k][k];

    for (j = k + 1; j < _PB_NJ; j++) {
      R[k][j] = 0;
      for (i = 0; i < _PB_NI; i++)
        R[k][j] += Q[i][k] * A[i][j];
      for (i = 0; i < _PB_NI; i++)
        A[i][j] = A[i][j] - Q[i][k] * R[k][j];
    }
  }
}

/* Array initialization. */
void init_array(int ni,
                int nj,
                DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj)) {
  int i, j;
  DATA_TYPE tmp;

  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) {
      tmp = frand();
      A[i][j] = tmp;
      tmp = 0;
      Q[i][j] = tmp;
    }
  }

  for (i = 0; i < nj; i++) {
    for (j = 0; j < nj; j++) {
      tmp = 0;
      R[i][j] = tmp;
    }
  }
}

void compareResults(int ni,
                    int nj,
                    DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                    DATA_TYPE POLYBENCH_2D(A_outputFromGpu, NI, NJ, ni, nj)) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++)
      if (percentDiff(A[i][j], A_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
        fail++;
  }

  // Print results
  printf(
    "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void gramschmidtCuda(int ni,
                     int nj,
                     DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                     DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                     DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj),
                     DATA_TYPE POLYBENCH_2D(A_outputFromGpu, NI, NJ, ni, nj),
                     DATA_TYPE POLYBENCH_2D(R_outputFromGpu, NJ, NJ, ni, nj),
                     DATA_TYPE POLYBENCH_2D(Q_outputFromGpu, NI, NJ, ni, nj)) {
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid1(1, 1);
  dim3 grid2((size_t) ceil(((float) NJ) / ((float) DIM_THREAD_BLOCK_X)), 1);
  dim3 grid3((size_t) ceil(((float) NJ) / ((float) DIM_THREAD_BLOCK_X)), 1);

  CUdeviceptr A_gpu;
  CUdeviceptr R_gpu;
  CUdeviceptr Q_gpu;

  checkCudaErrors(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NI * NJ));
  checkCudaErrors(cuMemAlloc(&R_gpu, sizeof(DATA_TYPE) * NJ * NJ));
  checkCudaErrors(cuMemAlloc(&Q_gpu, sizeof(DATA_TYPE) * NI * NJ));

  checkCudaErrors(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ));

  /* Start timer. */
  polybench_start_instruments;
  int k;
  for (k = 0; k < _PB_NJ; k++) {
    void* args1[6] = {&ni, &nj, &A_gpu, &R_gpu, &Q_gpu, &k};
    checkCudaErrors(
      cuLaunchKernel(kernels[0], grid1.x, grid1.y, grid1.z, block.x, block.y, block.z, 0, NULL, args1, NULL));
    checkCudaErrors(cuCtxSynchronize());

    void* args2[6] = {&ni, &nj, &A_gpu, &R_gpu, &Q_gpu, &k};
    checkCudaErrors(
      cuLaunchKernel(kernels[1], grid2.x, grid2.y, grid2.z, block.x, block.y, block.z, 0, NULL, args2, NULL));
    checkCudaErrors(cuCtxSynchronize());

    void* args3[6] = {&ni, &nj, &A_gpu, &R_gpu, &Q_gpu, &k};
    checkCudaErrors(
      cuLaunchKernel(kernels[2], grid3.x, grid3.y, grid3.z, block.x, block.y, block.z, 0, NULL, args3, NULL));
    checkCudaErrors(cuCtxSynchronize());
  }
  printf("GPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;

  checkCudaErrors(cuMemcpyDtoH(A_outputFromGpu, A_gpu, sizeof(DATA_TYPE) * NI * NJ));
  checkCudaErrors(cuMemcpyDtoH(R_outputFromGpu, R_gpu, sizeof(DATA_TYPE) * NJ * NJ));
  checkCudaErrors(cuMemcpyDtoH(Q_outputFromGpu, Q_gpu, sizeof(DATA_TYPE) * NI * NJ));

  checkCudaErrors(cuMemFree(A_gpu));
  checkCudaErrors(cuMemFree(R_gpu));
  checkCudaErrors(cuMemFree(Q_gpu));
  checkCudaErrors(cuCtxDestroy(cuContext));
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj)) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if (i % 20 == 0)
        fprintf(stderr, "\n");
    }

  fprintf(stderr, "\n");
}

int main(int argc, char* argv[]) {
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;

  ANN_A POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NJ, ni, nj);
  ANN_A POLYBENCH_2D_ARRAY_DECL(A_outputFromGpu, DATA_TYPE, NI, NJ, ni, nj);
  ANN_R POLYBENCH_2D_ARRAY_DECL(R, DATA_TYPE, NJ, NJ, nj, nj);
  ANN_Q POLYBENCH_2D_ARRAY_DECL(Q, DATA_TYPE, NI, NJ, ni, nj);
  ANN_R POLYBENCH_2D_ARRAY_DECL(R_outputFromGpu, DATA_TYPE, NJ, NJ, nj, nj);
  ANN_Q POLYBENCH_2D_ARRAY_DECL(Q_outputFromGpu, DATA_TYPE, NI, NJ, ni, nj);

  init_array(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));

  initCUDA(argc, argv);

  checkCudaErrors(cuModuleGetFunction(&(kernels[0]), cuModule, "gramschmidt_kernel1"));
  checkCudaErrors(cuModuleGetFunction(&(kernels[1]), cuModule, "gramschmidt_kernel2"));
  checkCudaErrors(cuModuleGetFunction(&(kernels[2]), cuModule, "gramschmidt_kernel3"));

  gramschmidtCuda(ni,
                  nj,
                  POLYBENCH_ARRAY(A),
                  POLYBENCH_ARRAY(R),
                  POLYBENCH_ARRAY(Q),
                  POLYBENCH_ARRAY(A_outputFromGpu),
                  POLYBENCH_ARRAY(R_outputFromGpu),
                  POLYBENCH_ARRAY(Q_outputFromGpu));

  /* Start timer. */
  polybench_start_instruments;

  // gramschmidt(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));

  printf("CPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;

  // compareResults(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));

  // print_array(ni, nj, POLYBENCH_ARRAY(A_outputFromGpu));
  print_array(ni, nj, POLYBENCH_ARRAY(Q_outputFromGpu));
  print_array(nj, nj, POLYBENCH_ARRAY(R_outputFromGpu));

  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(A_outputFromGpu);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);

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
