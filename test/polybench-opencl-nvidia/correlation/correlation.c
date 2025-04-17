/**
 * correlation.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "correlation.h"
#include "correlation_sh_ann.h"

#include <polybench.h>
#include <polybenchUtilFuncts.h>

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 2

#define MAX_SOURCE_SIZE (0x100000)

#define sqrt_of_array_cell(x, j) sqrt(x[j])

#if defined(cl_khr_fp64)   // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64) // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

char str_temp[1024];

// #define FLOAT_N 3214212.01
#define FLOAT_N N
#define EPS 0.005

cl_platform_id platform_id;
cl_device_id device_id;
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel_mean;
cl_kernel clKernel_std;
cl_kernel clKernel_reduce;
cl_kernel clKernel_corr;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem data_mem_obj;
cl_mem stddev_mem_obj;
cl_mem mean_mem_obj;
cl_mem symmat_mem_obj;
FILE* fp;
char* source_str;
size_t source_size;

// #define RUN_ON_CPU

void compareResults(int m,
                    int n,
                    DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n),
                    DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu, M, N, m, n)) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      // fprintf(stderr, "%d, %d, %lf, %lf\n", i, j, symmat[i][j], symmat_outputFromGpu[i][j]);
      DATA_TYPE a = symmat[i][j];
      DATA_TYPE b = symmat_outputFromGpu[i][j];
      if (percentDiff(a, b) > PERCENT_DIFF_ERROR_THRESHOLD)
        fail++;
    }
  }

  // print results
  printf(
    "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void read_cl_file() {
  // Load the kernel source code into the array source_str
#ifndef __TAFFO__
  fp = fopen("correlation.ptx", "r");
#else
  fp = fopen("correlation.taffo.ptx", "r");
#endif
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*) malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);
}

double frand(void) { return (double) rand() / (double) RAND_MAX; }

void init_arrays(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n)) {
  int i, j;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      DATA_TYPE d = (DATA_TYPE) frand();
      data[i][j] = (DATA_TYPE) d;
      // fprintf(stderr, "%f\n", data[i][j]);
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

void cl_mem_init(DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
                 DATA_TYPE POLYBENCH_1D(mean, M, m),
                 DATA_TYPE POLYBENCH_1D(stddev, M, m),
                 DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m)) {
  data_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
  symmat_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
  stddev_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M, NULL, &errcode);
  mean_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M, NULL, &errcode);

  if (errcode != CL_SUCCESS)
    printf("Error in creating buffers\n");

  errcode =
    clEnqueueWriteBuffer(clCommandQue, data_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M * N, data, 0, NULL, NULL);
  errcode =
    clEnqueueWriteBuffer(clCommandQue, symmat_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M * N, symmat, 0, NULL, NULL);
  errcode =
    clEnqueueWriteBuffer(clCommandQue, stddev_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M, stddev, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, mean_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M, mean, 0, NULL, NULL);
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
  clKernel_mean = clCreateKernel(clProgram, "mean_kernel", &errcode);
  if (errcode != CL_SUCCESS)
    printf("Error in creating kernel1\n");

  clKernel_std = clCreateKernel(clProgram, "std_kernel", &errcode);
  if (errcode != CL_SUCCESS)
    printf("Error in creating kernel2\n");

  clKernel_reduce = clCreateKernel(clProgram, "reduce_kernel", &errcode);
  if (errcode != CL_SUCCESS)
    printf("Error in creating kernel3\n");

  clKernel_corr = clCreateKernel(clProgram, "corr_kernel", &errcode);
  if (errcode != CL_SUCCESS)
    printf("Error in creating kernel4\n");
  clFinish(clCommandQue);
}

void cl_launch_kernel(int m, int n) {
  // FIXME: float_n is an array to work around the fact that scalars are ignored by the buffer_id mechanism
  DATA_TYPE ANN_FLOAT_N float_n[1];
  DATA_TYPE ANN_EPS eps[1];
  float_n[0] = FLOAT_N;
  eps[0] = EPS;

  size_t localWorkSize_Kernel1[2], globalWorkSize_Kernel1[2];
  size_t localWorkSize_Kernel2[2], globalWorkSize_Kernel2[2];
  size_t localWorkSize_Kernel3[2], globalWorkSize_Kernel3[2];
  size_t localWorkSize_Kernel4[2], globalWorkSize_Kernel4[2];

  localWorkSize_Kernel1[0] = DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
  localWorkSize_Kernel1[1] = DIM_LOCAL_WORK_GROUP_KERNEL_1_Y;
  globalWorkSize_Kernel1[0] =
    (size_t) ceil(((float) M) / ((float) DIM_LOCAL_WORK_GROUP_KERNEL_1_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
  globalWorkSize_Kernel1[1] = 1;

  localWorkSize_Kernel2[0] = DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
  localWorkSize_Kernel2[1] = DIM_LOCAL_WORK_GROUP_KERNEL_2_Y;
  globalWorkSize_Kernel2[0] =
    (size_t) ceil(((float) M) / ((float) DIM_LOCAL_WORK_GROUP_KERNEL_2_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
  globalWorkSize_Kernel2[1] = 1;

  localWorkSize_Kernel3[0] = DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
  localWorkSize_Kernel3[1] = DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;
  globalWorkSize_Kernel3[0] =
    (size_t) ceil(((float) M) / ((float) DIM_LOCAL_WORK_GROUP_KERNEL_3_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
  globalWorkSize_Kernel3[1] =
    (size_t) ceil(((float) N) / ((float) DIM_LOCAL_WORK_GROUP_KERNEL_3_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;

  localWorkSize_Kernel4[0] = DIM_LOCAL_WORK_GROUP_KERNEL_4_X;
  localWorkSize_Kernel4[1] = DIM_LOCAL_WORK_GROUP_KERNEL_4_Y;
  globalWorkSize_Kernel4[0] =
    (size_t) ceil(((float) M) / ((float) DIM_LOCAL_WORK_GROUP_KERNEL_4_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_4_X;
  globalWorkSize_Kernel4[1] = 1;

  /* Start timer. */
  polybench_start_instruments;

  // Set the arguments of the kernel
  errcode = clSetKernelArg(clKernel_mean, 0, sizeof(cl_mem), (void*) &mean_mem_obj);
  errcode |= clSetKernelArg(clKernel_mean, 1, sizeof(cl_mem), (void*) &data_mem_obj);
  errcode |= clSetKernelArg(clKernel_mean, 2, sizeof(DATA_TYPE), (void*) float_n);
  errcode |= clSetKernelArg(clKernel_mean, 3, sizeof(int), (void*) &m);
  errcode |= clSetKernelArg(clKernel_mean, 4, sizeof(int), (void*) &n);
  if (errcode != CL_SUCCESS)
    printf("Error in seting arguments1\n");

  // Execute the OpenCL kernel
  errcode = clEnqueueNDRangeKernel(
    clCommandQue, clKernel_mean, 1, NULL, globalWorkSize_Kernel1, localWorkSize_Kernel1, 0, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in launching kernel1\n");
  clEnqueueBarrier(clCommandQue);

  // Set the arguments of the kernel
  errcode = clSetKernelArg(clKernel_std, 0, sizeof(cl_mem), (void*) &mean_mem_obj);
  errcode = clSetKernelArg(clKernel_std, 1, sizeof(cl_mem), (void*) &stddev_mem_obj);
  errcode |= clSetKernelArg(clKernel_std, 2, sizeof(cl_mem), (void*) &data_mem_obj);
  errcode |= clSetKernelArg(clKernel_std, 3, sizeof(DATA_TYPE), (void*) float_n);
  errcode |= clSetKernelArg(clKernel_std, 4, sizeof(DATA_TYPE), (void*) eps);
  errcode |= clSetKernelArg(clKernel_std, 5, sizeof(int), (void*) &m);
  errcode |= clSetKernelArg(clKernel_std, 6, sizeof(int), (void*) &n);
  if (errcode != CL_SUCCESS)
    printf("Error in seting arguments2\n");

  // Execute the OpenCL kernel
  errcode = clEnqueueNDRangeKernel(
    clCommandQue, clKernel_std, 1, NULL, globalWorkSize_Kernel2, localWorkSize_Kernel2, 0, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in launching kernel2\n");
  clEnqueueBarrier(clCommandQue);

  // Set the arguments of the kernel
  errcode = clSetKernelArg(clKernel_reduce, 0, sizeof(cl_mem), (void*) &mean_mem_obj);
  errcode = clSetKernelArg(clKernel_reduce, 1, sizeof(cl_mem), (void*) &stddev_mem_obj);
  errcode |= clSetKernelArg(clKernel_reduce, 2, sizeof(cl_mem), (void*) &data_mem_obj);
  errcode |= clSetKernelArg(clKernel_reduce, 3, sizeof(DATA_TYPE), (void*) float_n);
  errcode |= clSetKernelArg(clKernel_reduce, 4, sizeof(int), (void*) &m);
  errcode |= clSetKernelArg(clKernel_reduce, 5, sizeof(int), (void*) &n);
  if (errcode != CL_SUCCESS)
    printf("Error in seting arguments3\n");

  // Execute the OpenCL kernel
  errcode = clEnqueueNDRangeKernel(
    clCommandQue, clKernel_reduce, 2, NULL, globalWorkSize_Kernel3, localWorkSize_Kernel3, 0, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in launching kernel3\n");
  clEnqueueBarrier(clCommandQue);

  // Set the arguments of the kernel
  errcode = clSetKernelArg(clKernel_corr, 0, sizeof(cl_mem), (void*) &symmat_mem_obj);
  errcode |= clSetKernelArg(clKernel_corr, 1, sizeof(cl_mem), (void*) &data_mem_obj);
  errcode |= clSetKernelArg(clKernel_corr, 2, sizeof(int), (void*) &m);
  errcode |= clSetKernelArg(clKernel_corr, 3, sizeof(int), (void*) &n);
  if (errcode != CL_SUCCESS)
    printf("Error in seting arguments4\n");

  // Execute the OpenCL kernel
  errcode = clEnqueueNDRangeKernel(
    clCommandQue, clKernel_corr, 1, NULL, globalWorkSize_Kernel4, localWorkSize_Kernel4, 0, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in launching kernel4\n");
  clEnqueueBarrier(clCommandQue);

  // How to fix an uninitialized value, polybench version...
  ANN_SYMMAT DATA_TYPE val[1];
  val[0] = 1.0;
  clEnqueueWriteBuffer(clCommandQue,
                       symmat_mem_obj,
                       CL_TRUE,
                       ((M - 1) * M + (M - 1)) * sizeof(DATA_TYPE),
                       sizeof(DATA_TYPE),
                       val,
                       0,
                       NULL,
                       NULL);

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
  errcode = clReleaseKernel(clKernel_reduce);
  errcode = clReleaseKernel(clKernel_mean);
  errcode = clReleaseKernel(clKernel_std);
  errcode = clReleaseKernel(clKernel_corr);
  errcode = clReleaseProgram(clProgram);
  errcode = clReleaseMemObject(symmat_mem_obj);
  errcode = clReleaseMemObject(data_mem_obj);
  errcode = clReleaseMemObject(mean_mem_obj);
  errcode = clReleaseMemObject(stddev_mem_obj);
  errcode = clReleaseCommandQueue(clCommandQue);
  errcode = clReleaseContext(clGPUContext);
  if (errcode != CL_SUCCESS)
    printf("Error in cleanup\n");
}

void correlation(int m,
                 int n,
                 DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
                 DATA_TYPE POLYBENCH_1D(mean, M, m),
                 DATA_TYPE POLYBENCH_1D(stddev, M, m),
                 DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n)) {
  int i, j, j1, j2;

  // Determine mean of column vectors of input data matrix
  for (j = 0; j < _PB_M; j++) {
    mean[j] = 0.0;

    for (i = 0; i < _PB_N; i++)
      mean[j] += data[i][j];

    mean[j] /= (DATA_TYPE) FLOAT_N;
    // fprintf(stderr, "%f\n", mean[j]);
  }

  // Determine standard deviations of column vectors of data matrix.
  for (j = 0; j < _PB_M; j++) {
    stddev[j] = 0.0;

    for (i = 0; i < _PB_N; i++)
      stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);

    stddev[j] /= FLOAT_N;
    stddev[j] = sqrt_of_array_cell(stddev, j);
    stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
    // fprintf(stderr, "%f\n", stddev[j]);
  }

  // Center and reduce the column vectors.
  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j < _PB_M; j++) {
      data[i][j] -= mean[j];
      __attribute__((annotate("scalar(range(-100, 100) final)"))) DATA_TYPE tmp = sqrt(FLOAT_N) * stddev[j];
      data[i][j] /= tmp;
      // fprintf(stderr, "%f\n", data[i][j]);
    }
  }

  // Calculate the m * m correlation matrix.
  for (j1 = 0; j1 < _PB_M - 1; j1++) {
    symmat[j1][j1] = 1.0;

    for (j2 = j1 + 1; j2 < _PB_M; j2++) {
      symmat[j1][j2] = 0.0;

      for (i = 0; i < _PB_N; i++)
        symmat[j1][j2] += (data[i][j1] * data[i][j2]);

      symmat[j2][j1] = symmat[j1][j2];
      // fprintf(stderr, "%f\n", symmat[j1][j2]);
    }
  }

  symmat[M - 1][M - 1] = 1.0;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int m, DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m))

{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * m + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

int main(int argc, char* argv[]) {
  int m = M;
  int n = N;

  ANN_DATA POLYBENCH_2D_ARRAY_DECL(data, DATA_TYPE, M, N, m, n);
  ANN_MEAN POLYBENCH_1D_ARRAY_DECL(mean, DATA_TYPE, M, m);
  ANN_STD POLYBENCH_1D_ARRAY_DECL(stddev, DATA_TYPE, M, m);
  ANN_SYMMAT POLYBENCH_2D_ARRAY_DECL(symmat, DATA_TYPE, M, N, m, n);
  ANN_SYMMAT POLYBENCH_2D_ARRAY_DECL(symmat_outputFromGpu, DATA_TYPE, M, N, m, n);
  // ANN_MEAN POLYBENCH_1D_ARRAY_DECL(mean_gpu,DATA_TYPE,M,m);
  // ANN_STD POLYBENCH_1D_ARRAY_DECL(stddev_gpu,DATA_TYPE,M,m);
  // ANN_DATA POLYBENCH_2D_ARRAY_DECL(data_gpu,DATA_TYPE,M,N,m,n);

  init_arrays(m, n, POLYBENCH_ARRAY(data));

  read_cl_file();
  cl_initialization();
  cl_mem_init(
    POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev), POLYBENCH_ARRAY(symmat_outputFromGpu));
  cl_load_prog();

  cl_launch_kernel(m, n);

  // errcode = clEnqueueReadBuffer(clCommandQue, data_mem_obj, CL_TRUE, 0, M * N * sizeof(DATA_TYPE),
  // POLYBENCH_ARRAY(data_gpu), 0, NULL, NULL); if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");
  errcode = clEnqueueReadBuffer(clCommandQue,
                                symmat_mem_obj,
                                CL_TRUE,
                                0,
                                M * N * sizeof(DATA_TYPE),
                                POLYBENCH_ARRAY(symmat_outputFromGpu),
                                0,
                                NULL,
                                NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in reading GPU mem\n");

#ifdef RUN_ON_CPU

  /* Start timer. */
  polybench_start_instruments;

  correlation(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev), POLYBENCH_ARRAY(symmat));

  /* Stop and print timer. */
  printf("CPU Time in seconds:\n");
  polybench_stop_instruments;
  polybench_print_instruments;

  compareResults(m, n, POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(symmat_outputFromGpu));

#endif // RUN_ON_CPU
  // for (int i=0; i<M; i++) {
  //	for (int j=0; j<N; j++)
  //		fprintf(stderr, "%e\n", data_gpu[i][j]);
  // }
  print_array(m, POLYBENCH_ARRAY(symmat_outputFromGpu));

  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);
  POLYBENCH_FREE_ARRAY(symmat);
  POLYBENCH_FREE_ARRAY(symmat_outputFromGpu);

  cl_clean_up();

  return 0;
}

#include <polybench.c>
