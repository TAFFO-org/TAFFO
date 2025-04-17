#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#define TRUE 1
#define FALSE 0
#include "data.h"
#ifndef M
#define M 10000
#endif

float ex0(float v) {
  const float p = 35000000.0f;
  const float a = 0.401f;
  const float b = 4.27e-05f;
  const float t = 300.0f;
  const float n = 1000.0f;
  const float k = 1.3806503e-23f;
  float pv = p * v;
  float __attribute__((annotate("scalar(range(401014, 3062257153))"))) ann = a * n * n / v;

  float pbb = p * b * b;

  float __attribute__((annotate("scalar(range(17123, 998543130625))"))) annnbvv = a * n * n * n * b / v / v;
  float __attribute__((annotate("scalar(range(-995480895487, 35382589.0))"))) ret = (pv + ann - pbb - annnbvv);
  return ret;
}

int main() {
  static const int len = sizeof(arr) / sizeof(arr[0]);
  float __attribute__((annotate("target('main') scalar(range(0, 1))"))) v[len];
  float res[len];
  for (int i = 0; i < len; ++i)
    v[i] = arr[i];

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
      res[j] = ex0(v[j]);

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
