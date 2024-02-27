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

float ex0(float radius, float theta)
{
  float pi = 3.14159265359f;
  float radiant = theta * (pi / 180.0f);
  return radius * sin(radiant);
}

int main()
{
  static const int len = sizeof(arr) / sizeof(arr[0]) / 2;
  float __attribute__((annotate("target('main') scalar(range(1, 10))")))
  radius[len];
  float __attribute__((annotate("scalar(range(0, 360))"))) theta[len];

  float res[len];
  for (int i = 0; i < len; ++i) {

    radius[i] = arr[i * 2];
    theta[i] = arr[i * 2 + 1];
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
      res[j] = ex0(radius[j], theta[j]);
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
