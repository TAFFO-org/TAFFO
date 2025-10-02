#include "common.h"
#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#define TRUE 1
#define FALSE 0
#include "data.h"
#ifndef M
#define M 1000
#endif

#ifdef APP_MFUNC
double sin(double x) { return x - ((x * x * x) / 6.0f); }

double __attribute__((annotate("scalar(range(-10, 10))"))) cos(double x) { return 1.0f - (x * x * 0.25f); }

double atan(double x) { return x - ((x * x * x) / 3.0f); }
#else
#include <math.h>
#endif

float __attribute__((annotate("scalar(range(-100, 100))"))) ex0(float radius, float theta) {
  float pi = 3.14159265359f;
  float __attribute__((annotate("scalar(range(-10, 10) type(64 54))"))) radiant = theta * (pi / 180.0f);
  float __attribute__((annotate("scalar(range(-100, 100))"))) c = cos(radiant);
  float __attribute__((annotate("scalar(range(-100, 100))"))) tmp = radius * c;
  return tmp;
}

int main() {
  static const int len = sizeof(arr) / sizeof(arr[0]) / 2;
  float __attribute__((annotate("target('main') scalar(range(1, 10))"))) radius[len];
  float __attribute__((annotate("scalar(range(0, 360))"))) theta[len];

  float __attribute__((annotate("scalar(range(-100, 100))"))) res[len];
  for (int i = 0; i < len; ++i) {

    radius[i] = arr[i * 2];
    theta[i] = arr[i * 2 + 1];
  }

  for (int i = 0; i < M; ++i) {
    uint32_t cycles_high1 = 0;
    uint32_t cycles_high = 0;
    uint32_t cycles_low = 0;
    uint32_t cycles_low1 = 0;

    CYCLES_START(cycles_high, cycles_low);
    for (int j = 0; j < len; ++j)
      res[j] = ex0(radius[j], theta[j]);

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
