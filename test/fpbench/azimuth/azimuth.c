#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#define TRUE 1
#define FALSE 0
#include "data.h"
#ifndef M
#define M 10
#endif

#ifdef APP_MFUNC
float sin(float x) { return x - ((x * x * x) / 6.0f); }

float cos(float x) { return 1.0f - (x * x * 0.25f); }

float atan(float x) { return x - ((x * x * x) / 3.0f); }
#else
#include <math.h>
#endif

float __attribute__((annotate("scalar(range(-176586304, 7394583.5) )")))
ex0(float lat1, float lat2, float lon1, float lon2) {
  float dLon = lon2 - lon1;
  float __attribute__((annotate("scalar(range(-1, 1) )"))) s_lat1 = sin(lat1);
  float __attribute__((annotate("scalar(range(-1, 1) )"))) c_lat1 = cos(lat1);
  float __attribute__((annotate("scalar(range(-1, 1) )"))) s_lat2 = sin(lat2);
  float __attribute__((annotate("scalar(range(-1, 1) )"))) c_lat2 = cos(lat2);
  float __attribute__((annotate("scalar(range(-100, 100) )"))) s_dLon = sin(dLon);
  float __attribute__((annotate("scalar(range(-100, 100) )"))) c_dLon = cos(dLon);
  float __attribute__((annotate("scalar(range(-100, 100) )"))) inner_atan =
    ((c_lat2 * s_dLon) / ((c_lat1 * s_lat2) - ((s_lat1 * c_lat2) * c_dLon)));
  float __attribute__((annotate("scalar(range(-176586304, 7394583.5) )"))) tmp = atan(inner_atan);
  return tmp;
}

int main() {
  static const int len = sizeof(arr) / sizeof(arr[0]) / 4;
  float __attribute__((annotate("scalar(range(-1, 1) ) "))) lat1[len];
  float __attribute__((annotate("scalar(range(-1, 1) )"))) lat2[len];
  float __attribute__((annotate("scalar(range(-4, 4) )"))) lon1[len];
  float __attribute__((annotate("scalar(range(-4, 4) )"))) lon2[len];
  float __attribute__((annotate("scalar(range(-176586304, 7394583.5) )"))) res[len];
  for (int i = 0; i < len; ++i) {

    lat1[i] = arr[i * 4];
    lat2[i] = arr[i * 4 + 1];
    lon1[i] = arr[i * 4 + 2];
    lon2[i] = arr[i * 4 + 3];
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
      res[j] = ex0(lat1[j], lat2[j], lon1[j], lon2[j]);

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
