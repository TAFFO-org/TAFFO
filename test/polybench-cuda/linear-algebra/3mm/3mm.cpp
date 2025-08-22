/**
 * 3mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "3mm.cuh"
#include "3mm_sh_ann.h"

#include <polybench.h>
#include <polybenchUtilFuncts.h>

#define GPU_DEVICE 0

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define RUN_ON_CPU

// define input ptx file
#ifndef PTX_FILE
#ifndef __TAFFO__
#define PTX_FILE "3mm.ptx"
#else
#define PTX_FILE "3mm.taffo.ptx"
#endif
#endif

static int initCUDA(int argc, char** argv);

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction kernels[3];
size_t totalGlobalMem;

const char* sSDKsample = "PolyBench 3mm (Driver API)";

void init_array(int ni,
                int nj,
                int nk,
                int nl,
                int nm,
                DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
                DATA_TYPE POLYBENCH_2D(C, NJ, NM, nj, nm),
                DATA_TYPE POLYBENCH_2D(D, NM, NL, nm, nl)) {
  __attribute__((annotate("scalar(range(0, 1000) final)"))) int i;
  __attribute__((annotate("scalar(range(0, 1000) final)"))) int j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i * j % ni) / (ni * 4);

  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i * (j + 1) % nj) / (nj * 4);

  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = (DATA_TYPE) (i * (j + 3) % nl) / (nl * 4);

  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) (i * (j + 2) % nk) / (nk * 4);
}

void compareResults(int ni,
                    int nl,
                    DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl),
                    DATA_TYPE POLYBENCH_2D(G_outputFromGpu, NI, NL, ni, nl)) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < ni; i++) {
    for (j = 0; j < nl; j++)
      if (percentDiff(G[i][j], G_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
        fail++;
  }

  // print results
  printf(
    "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

/* Main computational kernel on CPU */
void mm3_cpu(int ni,
             int nj,
             int nk,
             int nl,
             int nm,
             DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
             DATA_TYPE POLYBENCH_2D(B, NI, NK, ni, nk),
             DATA_TYPE POLYBENCH_2D(C, NK, NJ, nk, nj),
             DATA_TYPE POLYBENCH_2D(D, NJ, NL, nj, nl),
             DATA_TYPE POLYBENCH_2D(E, NJ, NM, nj, nm),
             DATA_TYPE POLYBENCH_2D(F, NM, NL, nm, nl),
             DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl)) {
  int i, j, k;

  /* E := A*B */
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NJ; j++) {
      E[i][j] = 0;
      for (k = 0; k < _PB_NK; ++k)
        E[i][j] += A[i][k] * B[k][j];
    }
  }

  /* F := C*D */
  for (i = 0; i < _PB_NJ; i++) {
    for (j = 0; j < _PB_NL; j++) {
      F[i][j] = 0;
      for (k = 0; k < _PB_NM; ++k)
        F[i][j] += C[i][k] * D[k][j];
    }
  }

  /* G := E*F */
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NL; j++) {
      G[i][j] = 0;
      for (k = 0; k < _PB_NJ; ++k)
        G[i][j] += E[i][k] * F[k][j];
    }
  }
}

void mm3Cuda(int ni,
             int nj,
             int nk,
             int nl,
             int nm,
             DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
             DATA_TYPE POLYBENCH_2D(B, NI, NK, ni, nk),
             DATA_TYPE POLYBENCH_2D(C, NK, NJ, nk, nj),
             DATA_TYPE POLYBENCH_2D(D, NJ, NL, nj, nl),
             DATA_TYPE POLYBENCH_2D(E, NJ, NM, nj, nm),
             DATA_TYPE POLYBENCH_2D(F, NM, NL, nm, nl),
             DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl),
             DATA_TYPE POLYBENCH_2D(G_outputFromGpu, NI, NL, ni, nl)) {
  CUdeviceptr A_gpu;
  CUdeviceptr B_gpu;
  CUdeviceptr C_gpu;
  CUdeviceptr D_gpu;
  CUdeviceptr E_gpu;
  CUdeviceptr F_gpu;
  CUdeviceptr G_gpu;

  checkCudaErrors(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NI * NK));
  checkCudaErrors(cuMemAlloc(&B_gpu, sizeof(DATA_TYPE) * NK * NJ));
  checkCudaErrors(cuMemAlloc(&C_gpu, sizeof(DATA_TYPE) * NJ * NM));
  checkCudaErrors(cuMemAlloc(&D_gpu, sizeof(DATA_TYPE) * NM * NL));
  checkCudaErrors(cuMemAlloc(&E_gpu, sizeof(DATA_TYPE) * NI * NJ));
  checkCudaErrors(cuMemAlloc(&F_gpu, sizeof(DATA_TYPE) * NJ * NL));
  checkCudaErrors(cuMemAlloc(&G_gpu, sizeof(DATA_TYPE) * NI * NL));

  checkCudaErrors(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NI * NK));
  checkCudaErrors(cuMemcpyHtoD(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ));
  checkCudaErrors(cuMemcpyHtoD(C_gpu, C, sizeof(DATA_TYPE) * NJ * NM));
  checkCudaErrors(cuMemcpyHtoD(D_gpu, D, sizeof(DATA_TYPE) * NM * NL));
  checkCudaErrors(cuMemcpyHtoD(E_gpu, E, sizeof(DATA_TYPE) * NI * NJ));
  checkCudaErrors(cuMemcpyHtoD(F_gpu, F, sizeof(DATA_TYPE) * NJ * NL));
  checkCudaErrors(cuMemcpyHtoD(G_gpu, G, sizeof(DATA_TYPE) * NI * NL));

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid1((size_t) (ceil(((float) NJ) / ((float) DIM_THREAD_BLOCK_X))),
             (size_t) (ceil((float) NI / ((float) DIM_THREAD_BLOCK_Y))));
  dim3 grid2((size_t) (ceil(((float) NL) / ((float) DIM_THREAD_BLOCK_X))),
             (size_t) (ceil((float) NJ / ((float) DIM_THREAD_BLOCK_Y))));
  dim3 grid3((size_t) (ceil(((float) NL) / ((float) DIM_THREAD_BLOCK_X))),
             (size_t) (ceil((float) NI / ((float) DIM_THREAD_BLOCK_Y))));

  /* Start timer. */
  polybench_start_instruments;

  void* args1[8] = {&ni, &nj, &nk, &nl, &nm, &A_gpu, &B_gpu, &E_gpu};
  checkCudaErrors(
    cuLaunchKernel(kernels[0], grid1.x, grid1.y, grid1.z, block.x, block.y, block.z, 0, NULL, args1, NULL));
  checkCudaErrors(cuCtxSynchronize());

  void* args2[8] = {&ni, &nj, &nk, &nl, &nm, &C_gpu, &D_gpu, &F_gpu};
  checkCudaErrors(
    cuLaunchKernel(kernels[1], grid2.x, grid2.y, grid2.z, block.x, block.y, block.z, 0, NULL, args2, NULL));
  checkCudaErrors(cuCtxSynchronize());

  void* args3[8] = {&ni, &nj, &nk, &nl, &nm, &E_gpu, &F_gpu, &G_gpu};
  checkCudaErrors(
    cuLaunchKernel(kernels[2], grid3.x, grid3.y, grid3.z, block.x, block.y, block.z, 0, NULL, args3, NULL));
  checkCudaErrors(cuCtxSynchronize());

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;
  checkCudaErrors(cuMemcpyDtoH(G_outputFromGpu, G_gpu, sizeof(DATA_TYPE) * NI * NL));

  checkCudaErrors(cuMemFree(A_gpu));
  checkCudaErrors(cuMemFree(B_gpu));
  checkCudaErrors(cuMemFree(C_gpu));
  checkCudaErrors(cuMemFree(D_gpu));
  checkCudaErrors(cuMemFree(E_gpu));
  checkCudaErrors(cuMemFree(F_gpu));
  checkCudaErrors(cuMemFree(G_gpu));
  checkCudaErrors(cuCtxDestroy(cuContext));
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nl, DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl)) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, G[i][j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

int main(int argc, char** argv) {
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;

  /* Variable declaration/allocation. */
  ANN_A POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  ANN_B POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  ANN_C POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
  ANN_D POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
  ANN_E POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
  ANN_F POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
  ANN_G POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);
  ANN_G POLYBENCH_2D_ARRAY_DECL(G_outputFromGpu, DATA_TYPE, NI, NL, ni, nl);

  init_array(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

  initCUDA(argc, argv);

  checkCudaErrors(cuModuleGetFunction(&(kernels[0]), cuModule, "mm3_kernel1"));
  checkCudaErrors(cuModuleGetFunction(&(kernels[1]), cuModule, "mm3_kernel2"));
  checkCudaErrors(cuModuleGetFunction(&(kernels[2]), cuModule, "mm3_kernel3"));

  mm3Cuda(ni,
          nj,
          nk,
          nl,
          nm,
          POLYBENCH_ARRAY(A),
          POLYBENCH_ARRAY(B),
          POLYBENCH_ARRAY(C),
          POLYBENCH_ARRAY(D),
          POLYBENCH_ARRAY(E),
          POLYBENCH_ARRAY(F),
          POLYBENCH_ARRAY(G),
          POLYBENCH_ARRAY(G_outputFromGpu));

  /* Start timer. */
  polybench_start_instruments;

  // mm3_cpu(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D),
  // POLYBENCH_ARRAY(E), 	POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(G));

  /* Stop and print timer. */
  printf("CPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;

  // compareResults(ni, nl, POLYBENCH_ARRAY(G), POLYBENCH_ARRAY(G_outputFromGpu));

  print_array(ni, nl, POLYBENCH_ARRAY(G_outputFromGpu));

  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(E);
  POLYBENCH_FREE_ARRAY(F);
  POLYBENCH_FREE_ARRAY(G);
  POLYBENCH_FREE_ARRAY(G_outputFromGpu);

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
