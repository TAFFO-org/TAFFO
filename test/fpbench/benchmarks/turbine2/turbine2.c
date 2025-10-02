#include "common.h"
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
float ex0(float v, float w, float r) {
  return ((3.0f + (2.0f / (r * r))) - (((0.125f * (3.0f - (2.0f * v))) * (((w * w) * r) * r)) / (1.0f - v))) - 4.5f;
}

int main() {
  static const int len = sizeof(arr) / sizeof(arr[0]) / 3;
  float __attribute__((annotate("target('main') scalar(range(-5, 0))"))) v[len];
  float __attribute__((annotate("scalar(range(0, 1))"))) w[len];
  float __attribute__((annotate("scalar(range(3.8, 7.8))"))) e[len];

  float res[len];

  for (int i = 0; i < len; ++i) {
    v[i] = arr[i * 3];
    w[i] = arr[i * 3 + 1];
    e[i] = arr[i * 3 + 2];
  }

  for (int i = 0; i < M; ++i) {
    uint32_t cycles_high1 = 0;
    uint32_t cycles_high = 0;
    uint32_t cycles_low = 0;
    uint32_t cycles_low1 = 0;

    CYCLES_START(cycles_high, cycles_low);
    for (int j = 0; j < len; ++j)
      res[j] = ex0(v[j], w[j], e[j]);

    CYCLES_END(cycles_high1, cycles_low1);
    uint64_t end = (uint64_t) cycles_high1 << 32 | cycles_low1;
    uint64_t start = (uint64_t) cycles_high << 32 | cycles_low;
    if (end > start)
      printf("Cycles: %li\n", end - start);
  }
  printf("Values Begin\n");
  for (int j = 0; j < len; ++j)
    printf("%.16f\n", res[j]);
  printf("Values End\n");
  return 0;
}
