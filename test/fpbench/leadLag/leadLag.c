#include "stdio.h"
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0
#include "data.h"
#ifndef M
#define M 100
#endif

float ex0(float y, float yd) {
  float eps = 0.01f;
  float Dc = -1280.0f;
  float Ac00 = 0.499f;
  float Ac01 = -0.05f;
  float Ac10 = 0.01f;
  float Ac11 = 1.0f;
  float Bc0 = 1.0f;
  float Bc1 = 0.0f;
  float Cc0 = 564.48f;
  float Cc1 = 0.0f;
  float yc = 0.0f;
  float u = 0.0f;
  float xc0 = 0.0f;
  float xc1 = 0.0f;
  float i = 0.0f;
  float e = 1.0f;
  int tmp = e > eps;
  while (tmp) {
    float v = y - yd;
    float tmp_1;
    if (v < -1.0f) {
      tmp_1 = -1.0f;
    } else if (1.0f < v) {
      tmp_1 = 1.0f;
    } else {
      tmp_1 = v;
    }
    yc = tmp_1;
    u = (Cc0 * xc0) + ((Cc1 * xc1) + (Dc * yc));
    xc0 = (Ac00 * xc0) + ((Ac01 * xc1) + (Bc0 * yc));
    xc1 = (Ac10 * xc0) + ((Ac11 * xc1) + (Bc1 * yc));
    i = i + 1.0f;
    e = fabsf((yc - xc1));
    tmp = e > eps;
  }
  return xc1;
}

int main() {
  static const int len = sizeof(arr) / sizeof(arr[0]) / 2;
  float __attribute__((annotate("target('main') scalar(range(0, 50))"))) y[len];
  float __attribute__((annotate("scalar(range(0, 50))"))) yd[len];

  float res[len];
  for (int i = 0; i < len; ++i) {

    y[i] = arr[i * 2];
    yd[i] = arr[i * 2 + 1];
  }

  for (int i = 0; i < M; ++i) {
    uint32_t cycles_high1 = 0;
    uint32_t cycles_high = 0;
    uint32_t cycles_low = 0;
    uint32_t cycles_low1 = 0;

    asm volatile("CPUID\n\t"
                 "RDTSC\n\t"
                 "mov %%edx, %0\n\t"
                 "mov %%eax, %1\n\t"
                 : "=r"(cycles_high), "=r"(cycles_low)::"%rax", "%rbx", "%rcx",
                   "%rdx");
    for (int j = 0; j < len; ++j) {
      res[j] = ex0(y[j], yd[j]);
    }

    asm volatile("RDTSCP\n\t"
                 "mov %%edx, %0\n\t"
                 "mov %%eax, %1\n\t"
                 "CPUID\n\t"
                 : "=r"(cycles_high1), "=r"(cycles_low1)::"%rax", "%rbx",
                   "%rcx", "%rdx");
    uint64_t end = (uint64_t)cycles_high1 << 32 | cycles_low1;
    uint64_t start = (uint64_t)cycles_high << 32 | cycles_low;
    if (end > start) {
      printf("Cycles: %li\n", end - start);
    }
  }
  printf("Values Begin\n");
  for (int j = 0; j < len; ++j) {
    printf("%f\n", res[j]);
  }
  printf("Values End\n");
  return 0;
}
