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
float ex0(float v, float w, float r)
{
  return ((3.0f + (2.0f / (r * r))) -
          (((0.125f * (3.0f - (2.0f * v))) * (((w * w) * r) * r)) / (1.0f - v))) -
         4.5f;
}

int internal_main()
{
  static const int len = sizeof(arr) / sizeof(arr[0]) / 3;
  float __attribute__((annotate("target('main') scalar(range(-5, 0) final)")))
  v[len];
  float __attribute__((annotate("scalar(range(0, 1) final)"))) w[len];
  float __attribute__((annotate("scalar(range(3.8, 7.8) final)"))) e[len];

  float res[len];

  for (int i = 0; i < len; ++i) {
    v[i] = arr[i * 3];
    w[i] = arr[i * 3 + 1];
    e[i] = arr[i * 3 + 2];
  }

  for (int i = 0; i < M; ++i) {


    long long start = miosix::getTime();
    for (int j = 0; j < len; ++j) {
      res[j] = ex0(v[j], w[j], e[j]);
    }

    long long end = miosix::getTime();

    if (end > start) {
      printf("Cycles: %lli\n", end - start);
    }
  }
  printf("Values Begin\n");
  for (int j = 0; j < len; ++j) {
    printf("%.16f\n", res[j]);
  }
  printf("Values End\n");
  return 0;
}