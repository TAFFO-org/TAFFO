#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0
#include "data.h"
#include "stdio.h"
#ifndef M
#define M 10000
#endif

float ex0(float a, float b, float c)
{
  float s = ((a + b) + c) / 2.0;
  return sqrt((((s * (s - a)) * (s - b)) * (s - c)));
}

int internal_main()
{
  static const int len = sizeof(arr) / sizeof(arr[0]) / 3;
  float __attribute__((annotate("target('main') scalar(range(-9, 9) final)")))
  a[len];
  float __attribute__((annotate("scalar(range(-9, -9) final)"))) b[len];
  float __attribute__((annotate("scalar(range(-9, 9) final)"))) c[len];

  float res[len];

  for (size_t i = 0; i < len; ++i) {
    a[i] = arr[i * 3];
    b[i] = arr[i * 3 + 1];
    c[i] = arr[i * 3 + 2];
  }

  for (int i = 0; i < M; ++i) {


    long long start = miosix::getTime();
    for (int j = 0; j < len; ++j) {
      res[j] = ex0(a[j], b[j], c[j]);
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