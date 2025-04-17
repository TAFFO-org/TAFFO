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

int internal_main() {
  static const int len = sizeof(arr) / sizeof(arr[0]) / 2;
  float __attribute__((annotate("target('main') scalar(range(-5, 5))"))) x1[len];
  float __attribute__((annotate("scalar(range(-20, 5))"))) x2[len];

  float res[len];

  for (int i = 0; i < len; ++i) {
    x1[i] = arr[i * 2];
    x2[i] = arr[i * 2 + 1];
  }

  for (int i = 0; i < M; ++i) {

    long long start = miosix::getTime();
    for (int j = 0; j < len; ++j)
      res[j] = ex0(x1[j], x2[j]);

    long long end = miosix::getTime();

    if (end > start)
      printf("Cycles: %lli\n", end - start);
  }
  printf("Values Begin\n");
  for (int j = 0; j < len; ++j)
    printf("%f\n", res[j]);
  printf("Values End\n");
  return 0;
}
