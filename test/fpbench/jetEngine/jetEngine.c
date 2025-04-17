#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#define TRUE 1
#define FALSE 0
#include "data.h"
#ifndef M
#define M 10000
#endif

float ex0(float x1, float x2) {
  float t = (((3.0f * x1) * x1) + (2.0f * x2)) - x1;
  float t_42_ = (((3.0f * x1) * x1) - (2.0f * x2)) - x1;
  float d = (x1 * x1) + 1.0f;
  float s = t / d;
  float s_42_ = t_42_ / d;
  return x1
       + (((((((((2.0f * x1) * s) * (s - 3.0f)) + ((x1 * x1) * ((4.0f * s) - 6.0f))) * d) + (((3.0f * x1) * x1) * s))
            + ((x1 * x1) * x1))
           + x1)
          + (3.0f * s_42_));
}

int main() {
  static const int len = sizeof(arr) / sizeof(arr[0]) / 2;
  float __attribute__((annotate("target('main') scalar(range(-5, 5))"))) x1[len];
  float __attribute__((annotate("scalar(range(-20, 5))"))) x2[len];

  float res[len];

  for (int i = 0; i < len; ++i) {
    x1[i] = arr[i * 2];
    x2[i] = arr[i * 2 + 1];
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
                 : "=r"(cycles_high), "=r"(cycles_low)::"%rax", "%rbx", "%rcx", "%rdx");
    for (int j = 0; j < len; ++j)
      res[j] = ex0(x1[j], x2[j]);

    asm volatile("RDTSCP\n\t"
                 "mov %%edx, %0\n\t"
                 "mov %%eax, %1\n\t"
                 "CPUID\n\t"
                 : "=r"(cycles_high1), "=r"(cycles_low1)::"%rax", "%rbx", "%rcx", "%rdx");
    uint64_t end = (uint64_t) cycles_high1 << 32 | cycles_low1;
    uint64_t start = (uint64_t) cycles_high << 32 | cycles_low;
    if (end > start)
      printf("Cycles: %li\n", end - start);
  }
  printf("Values Begin\n");
  for (int j = 0; j < len; ++j)
    printf("%f\n", res[j]);
  printf("Values End\n");
  return 0;
}
