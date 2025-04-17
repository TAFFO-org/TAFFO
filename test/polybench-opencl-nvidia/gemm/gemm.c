/**
 * gemm.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define POLYBENCH_TIME 1

// select the OpenCL device to use (can be GPU, CPU, or Accelerator such as Intel Xeon Phi)
#define OPENCL_DEVICE_SELECTION CL_DEVICE_TYPE_GPU

#include "gemm.h"
#include "gemm_sh_ann.h"

#include <polybench.h>
#include <polybenchUtilFuncts.h>

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

#if defined(cl_khr_fp64)   // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64) // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

char str_temp[1024];

cl_platform_id platform_id;
cl_device_id device_id;
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem a_mem_obj;
cl_mem b_mem_obj;
cl_mem c_mem_obj;
FILE* fp;
char* source_str;
size_t source_size;

// #define RUN_ON_CPU

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

void read_cl_file() {
  // Load the kernel source code into the array source_str
#ifndef __TAFFO__
  fp = fopen("gemm.ptx", "r");
#else
  fp = fopen("gemm.taffo.ptx", "r");
#endif
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*) malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);
}

void init(int ni,
          int nj,
          int nk,
          DATA_TYPE* alpha,
          DATA_TYPE* beta,
          DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
          DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
          DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj)) {
  int i, j;
  float tmp;

  *alpha = 32412 / 115;
  *beta = 2123 / 115;

  for (i = 0; i < ni; i++) {
    for (j = 0; j < nk; j++) {
      tmp = ((DATA_TYPE) i * j) / (NI * NK);
      A[i][j] = tmp;
    }
  }

  for (i = 0; i < nk; i++) {
    for (j = 0; j < nj; j++) {
      tmp = ((DATA_TYPE) i * j) / (NK * NJ);
      B[i][j] = tmp;
    }
  }

  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) {
      tmp = ((DATA_TYPE) i * j) / (NI * NJ);
      C[i][j] = tmp;
    }
  }
}

void cl_initialization() {
  // Get platform and device information
  errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
  if (errcode == CL_SUCCESS)
    printf("number of platforms is %d\n", num_platforms);
  else
    printf("Error getting platform IDs\n");

  errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(str_temp), str_temp, NULL);
  if (errcode == CL_SUCCESS)
    printf("platform name is %s\n", str_temp);
  else
    printf("Error getting platform name\n");

  errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp, NULL);
  if (errcode == CL_SUCCESS)
    printf("platform version is %s\n", str_temp);
  else
    printf("Error getting platform version\n");

  errcode = clGetDeviceIDs(platform_id, OPENCL_DEVICE_SELECTION, 1, &device_id, &num_devices);
  if (errcode == CL_SUCCESS)
    printf("number of devices is %d\n", num_devices);
  else
    printf("Error getting device IDs\n");

  errcode = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(str_temp), str_temp, NULL);
  if (errcode == CL_SUCCESS)
    printf("device name is %s\n", str_temp);
  else
    printf("Error getting device name\n");

  // Create an OpenCL context
  clGPUContext = clCreateContext(NULL, 1, &device_id, NULL, NULL, &errcode);
  if (errcode != CL_SUCCESS)
    printf("Error in creating context\n");

  // Create a command-queue
  clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
  if (errcode != CL_SUCCESS)
    printf("Error in creating command queue\n");
}

void cl_mem_init(DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                 DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
                 DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj)) {
  a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NK, NULL, &errcode);
  b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NK * NJ, NULL, &errcode);
  c_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NJ, NULL, &errcode);

  if (errcode != CL_SUCCESS)
    printf("Error in creating buffers\n");

  errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NK, A, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NK * NJ, B, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ, C, 0, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in writing buffers\n");
}

void cl_load_prog() {
  // Create a program from the kernel source
  clProgram = clCreateProgramWithBinary(
    clGPUContext, 1, &device_id, (const size_t*) &source_size, (const char**) &source_str, NULL, &errcode);

  if (errcode != CL_SUCCESS)
    printf("Error in creating program\n");

  // Build the program
  errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in building program\n");

  // Create the OpenCL kernel
  clKernel = clCreateKernel(clProgram, "gemm", &errcode);
  if (errcode != CL_SUCCESS)
    printf("Error in creating kernel\n");
  clFinish(clCommandQue);
}

void cl_launch_kernel(int ni, int nj, int nk, DATA_TYPE* alpha, DATA_TYPE* beta) {
  size_t localWorkSize[2], globalWorkSize[2];
  localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
  globalWorkSize[0] = (size_t) ceil(((float) NJ) / ((float) DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
  globalWorkSize[1] = (size_t) ceil(((float) NI) / ((float) DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

  /* Start timer. */
  polybench_start_instruments;

  // Set the arguments of the kernel
  errcode = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void*) &a_mem_obj);
  if (errcode != CL_SUCCESS)
    printf("Error in setting argument a_mem_obj\n");
  errcode = clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void*) &b_mem_obj);
  if (errcode != CL_SUCCESS)
    printf("Error in setting argument b_mem_obj\n");
  errcode = clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void*) &c_mem_obj);
  if (errcode != CL_SUCCESS)
    printf("Error in setting argument c_mem_obj\n");
  errcode = clSetKernelArg(clKernel, 3, sizeof(DATA_TYPE), (void*) alpha);
  if (errcode != CL_SUCCESS)
    printf("Error in setting argument alpha\n");
  errcode = clSetKernelArg(clKernel, 4, sizeof(DATA_TYPE), (void*) beta);
  if (errcode != CL_SUCCESS)
    printf("Error in setting argument beta\n");
  errcode = clSetKernelArg(clKernel, 5, sizeof(int), (void*) &ni);
  if (errcode != CL_SUCCESS)
    printf("Error in setting argument ni\n");
  errcode = clSetKernelArg(clKernel, 6, sizeof(int), (void*) &nj);
  if (errcode != CL_SUCCESS)
    printf("Error in setting argument nj\n");
  errcode = clSetKernelArg(clKernel, 7, sizeof(int), (void*) &nk);
  if (errcode != CL_SUCCESS)
    printf("Error in setting argument nk\n");

  // Execute the OpenCL kernel
  errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in launching kernel\n");
  clFinish(clCommandQue);

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;
}

void cl_clean_up() {
  // Clean up
  errcode = clFlush(clCommandQue);
  errcode = clFinish(clCommandQue);
  errcode = clReleaseKernel(clKernel);
  errcode = clReleaseProgram(clProgram);
  errcode = clReleaseMemObject(a_mem_obj);
  errcode = clReleaseMemObject(b_mem_obj);
  errcode = clReleaseMemObject(c_mem_obj);
  errcode = clReleaseCommandQueue(clCommandQue);
  errcode = clReleaseContext(clGPUContext);
  if (errcode != CL_SUCCESS)
    printf("Error in cleanup\n");
}

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
  ANN_ALPHA DATA_TYPE alpha[1];
  ANN_BETA DATA_TYPE beta[1];
  ANN_A POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  ANN_B POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  ANN_C POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NI, NJ, ni, nj);
  ANN_C POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu, DATA_TYPE, NI, NJ, ni, nj);

  init(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

  read_cl_file();
  cl_initialization();
  cl_mem_init(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));
  cl_load_prog();

  cl_launch_kernel(ni, nj, nk, alpha, beta);

  errcode = clEnqueueReadBuffer(
    clCommandQue, c_mem_obj, CL_TRUE, 0, NI * NJ * sizeof(DATA_TYPE), POLYBENCH_ARRAY(C_outputFromGpu), 0, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in reading GPU mem\n");

#ifdef RUN_ON_CPU

  /* Start timer. */
  polybench_start_instruments;

  gemm(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

  /* Stop and print timer. */
  printf("CPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;

  compareResults(ni, nj, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

#else  // prevent dead code elimination

  polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(C_outputFromGpu)));

#endif // RUN_ON_CPU
  print_array(ni, nj, POLYBENCH_ARRAY(C_outputFromGpu));

  cl_clean_up();

  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(C_outputFromGpu);

  return 0;
}

#include <polybench.c>
