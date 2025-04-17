/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "atax.cuh"
#include "atax_sh_ann.h"

#include <polybench.h>
#include <polybenchUtilFuncts.h>

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
#define PTX_FILE "atax.ptx"
#else
#define PTX_FILE "atax.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char** argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernels[2];
size_t totalGlobalMem;

const char* sSDKsample = "PolyBench atax (Driver API)";

void init_array(int nx, int ny, DATA_TYPE POLYBENCH_1D(x, NX, nx), DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny)) {
  __attribute__((annotate("scalar(range(0, 10000) final)"))) int i;
  __attribute__((annotate("scalar(range(0, 10000) final)"))) int j;

  __attribute__((annotate("scalar(range(0, 10000) final)"))) DATA_TYPE nf = NX;

  for (i = 0; i < nx; i++) {
    x[i] = (DATA_TYPE) (i / nf);
    for (j = 0; j < ny; j++)
      A[i][j] = (DATA_TYPE) (i * j % NX) / (NX * 7);
  }
}

void compareResults(int ny, DATA_TYPE POLYBENCH_1D(z, NY, ny), DATA_TYPE POLYBENCH_1D(z_outputFromGpu, NY, ny)) {
  int i, fail;
  fail = 0;

  for (i = 0; i < ny; i++)
    if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
      fail++;

  // print results
  printf(
    "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void atax_cpu(int nx,
              int ny,
              DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny),
              DATA_TYPE POLYBENCH_1D(x, NY, ny),
              DATA_TYPE POLYBENCH_1D(y, NY, ny),
              DATA_TYPE POLYBENCH_1D(tmp, NX, nx)) {
  int i, j;

  ANN_TMP DATA_TYPE max_tmp = 0;
  ANN_Y DATA_TYPE max_y = 0;
  for (i = 0; i < _PB_NY; i++)
    y[i] = 0;
  for (i = 0; i < _PB_NX; i++) {
    tmp[i] = 0;

    for (j = 0; j < _PB_NY; j++) {
      tmp[i] = tmp[i] + A[i][j] * x[j];
      if (tmp[i] > max_tmp)
        max_tmp = tmp[i];
    }

    for (j = 0; j < _PB_NY; j++) {
      y[j] = y[j] + A[i][j] * tmp[i];
      if (y[j] > max_y)
        max_y = y[j];
    }
  }
  printf("Max tmp and y: %lf %lf \n", max_tmp, max_y);
}

void ataxGpu(int nx,
             int ny,
             DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny),
             DATA_TYPE POLYBENCH_1D(x, NX, nx),
             DATA_TYPE POLYBENCH_1D(y, NY, ny),
             DATA_TYPE POLYBENCH_1D(tmp, NX, nx),
             DATA_TYPE POLYBENCH_1D(y_outputFromGpu, NY, ny)) {
  CUdeviceptr A_gpu;
  CUdeviceptr x_gpu;
  CUdeviceptr y_gpu;
  CUdeviceptr tmp_gpu;

  checkCudaErrors(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NX * NY));
  checkCudaErrors(cuMemAlloc(&x_gpu, sizeof(DATA_TYPE) * NY));
  checkCudaErrors(cuMemAlloc(&y_gpu, sizeof(DATA_TYPE) * NY));
  checkCudaErrors(cuMemAlloc(&tmp_gpu, sizeof(DATA_TYPE) * NX));

  checkCudaErrors(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NX * NY));
  checkCudaErrors(cuMemcpyHtoD(x_gpu, x, sizeof(DATA_TYPE) * NY));
  checkCudaErrors(cuMemcpyHtoD(y_gpu, y, sizeof(DATA_TYPE) * NY));
  checkCudaErrors(cuMemcpyHtoD(tmp_gpu, tmp, sizeof(DATA_TYPE) * NX));

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid1((size_t) (ceil(((float) NX) / ((float) block.x))), 1);
  dim3 grid2((size_t) (ceil(((float) NY) / ((float) block.x))), 1);

  /* Start timer. */
  polybench_start_instruments;

  void* args1[5] = {&nx, &ny, &A_gpu, &x_gpu, &tmp_gpu};
  checkCudaErrors(
    cuLaunchKernel(kernels[0], grid1.x, grid1.y, grid1.z, block.x, block.y, block.z, 0, NULL, args1, NULL));
  checkCudaErrors(cuCtxSynchronize());

  void* args2[5] = {&nx, &ny, &A_gpu, &y_gpu, &tmp_gpu};
  checkCudaErrors(
    cuLaunchKernel(kernels[1], grid2.x, grid2.y, grid2.z, block.x, block.y, block.z, 0, NULL, args2, NULL));
  checkCudaErrors(cuCtxSynchronize());

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;

  checkCudaErrors(cuMemcpyDtoH(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * NX));

  checkCudaErrors(cuMemFree(A_gpu));
  checkCudaErrors(cuMemFree(x_gpu));
  checkCudaErrors(cuMemFree(y_gpu));
  checkCudaErrors(cuMemFree(tmp_gpu));
  checkCudaErrors(cuCtxDestroy(cuContext));
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nx, DATA_TYPE POLYBENCH_1D(y, NX, nx)) {
  int i;

  for (i = 0; i < nx; i++) {
    fprintf(stderr, DATA_PRINTF_MODIFIER, y[i]);
    if (i % 20 == 0)
      fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

int main(int argc, char** argv) {
  int nx = NX;
  int ny = NY;

  ANN_A POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny);
  ANN_X POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, NY, ny);
  ANN_Y POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, NY, ny);
  ANN_Y POLYBENCH_1D_ARRAY_DECL(y_outputFromGpu, DATA_TYPE, NY, ny);
  ANN_TMP POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, NX, nx);

  init_array(nx, ny, POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(A));

  initCUDA(argc, argv);

  checkCudaErrors(cuModuleGetFunction(&(kernels[0]), cuModule, "atax_kernel1"));
  checkCudaErrors(cuModuleGetFunction(&(kernels[1]), cuModule, "atax_kernel2"));

  ataxGpu(nx,
          ny,
          POLYBENCH_ARRAY(A),
          POLYBENCH_ARRAY(x),
          POLYBENCH_ARRAY(y),
          POLYBENCH_ARRAY(tmp),
          POLYBENCH_ARRAY(y_outputFromGpu));

  /* Start timer. */
  polybench_start_instruments;

  atax_cpu(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(tmp));

  /* Stop and print timer. */
  printf("CPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;

  // compareResults(ny, POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(y_outputFromGpu));

  print_array(ny, POLYBENCH_ARRAY(y_outputFromGpu));

  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(y_outputFromGpu);
  POLYBENCH_FREE_ARRAY(tmp);

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
