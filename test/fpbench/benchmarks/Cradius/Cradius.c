#include "common.h"
#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#define TRUE 1
#define FALSE 0
#include "data.h"
#ifndef M
#define M 10000
#endif

#ifdef APP_MFUNC

#else
#include <math.h>

#endif

float ex0(float x, float y) { return sqrt((x * x) + (y * y)); }

int main() {
  static const int len = sizeof(arr) / sizeof(arr[0]) / 2;
  float __attribute__((annotate("target('main') scalar(range(1, 100))"))) x[len];
  float __attribute__((annotate("scalar(range(1, 100))"))) y[len];

  float res[len];
  for (int i = 0; i < len; ++i) {

    x[i] = arr[i * 2];
    y[i] = arr[i * 2 + 1];
  }

  for (int i = 0; i < M; ++i) {
    uint32_t cycles_high1 = 0;
    uint32_t cycles_high = 0;
    uint32_t cycles_low = 0;
    uint32_t cycles_low1 = 0;

    CYCLES_START(cycles_high, cycles_low);
    for (int j = 0; j < len; ++j)
      res[j] = ex0(x[j], y[j]);

    CYCLES_END(cycles_high1, cycles_low1);
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
