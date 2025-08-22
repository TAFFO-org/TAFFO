#include <miosix.h>
#define TRUE 1
#define FALSE 0
#include "data.h"
#ifndef M
#define M 10000
#endif

#ifdef APP_MFUNC
float sin(float x) { return x - ((x * x * x) / 6.0f); }

float cos(float x) { return 1.0f - (x * x * 0.25f); }

float atan(float x) { return x - ((x * x * x) / 3.0f); }
#else
#include <math.h>
#endif

float __attribute__((annotate("scalar(range(-176586304, 7394583.5) type(64 35) final)")))
ex0(float lat1, float lat2, float lon1, float lon2) {
  float dLon = lon2 - lon1;
  float __attribute__((annotate("scalar(range(-1, 1) type(64 56) final)"))) s_lat1 = sin(lat1);
  float __attribute__((annotate("scalar(range(-1, 1) type(64 56) final)"))) c_lat1 = cos(lat1);
  float __attribute__((annotate("scalar(range(-1, 1) type(64 56) final)"))) s_lat2 = sin(lat2);
  float __attribute__((annotate("scalar(range(-1, 1) type(64 56) final)"))) c_lat2 = cos(lat2);
  float __attribute__((annotate("scalar(range(-100, 100) type(64 56) final)"))) s_dLon = sin(dLon);
  float __attribute__((annotate("scalar(range(-100, 100) type(64 56) final)"))) c_dLon = cos(dLon);
  float __attribute__((annotate("scalar(range(-100, 100) type(64 35) final)"))) inner_atan =
    ((c_lat2 * s_dLon) / ((c_lat1 * s_lat2) - ((s_lat1 * c_lat2) * c_dLon)));
  float __attribute__((annotate("scalar(range(-176586304, 7394583.5) type(64 35) final)"))) tmp = atan(inner_atan);
  return tmp;
}

int internal_main() {
  static const int len = sizeof(arr) / sizeof(arr[0]) / 4;
  float __attribute__((annotate("scalar(range(-1, 1) type(64 60) final) "))) lat1[len];
  float __attribute__((annotate("scalar(range(-1, 1) type(64 60) final)"))) lat2[len];
  float __attribute__((annotate("scalar(range(-4, 4) type(64 60) final)"))) lon1[len];
  float __attribute__((annotate("scalar(range(-4, 4) type(64 60) final)"))) lon2[len];
  float __attribute__((annotate("scalar(range(-176586304, 7394583.5) type(64 35) final)"))) res[len];
  for (int i = 0; i < len; ++i) {
    lat1[i] = arr[i * 4];
    lat2[i] = arr[i * 4 + 1];
    lon1[i] = arr[i * 4 + 2];
    lon2[i] = arr[i * 4 + 3];
  }

  for (int i = 0; i < M; ++i) {

    long long start = miosix::getTime();
    for (int j = 0; j < len; ++j)
      res[j] = ex0(lat1[j], lat2[j], lon1[j], lon2[j]);

    long long end = miosix::getTime();

    printf("Cycles: %lli\n", end - start);
  }
  printf("Values Begin\n");
  for (int j = 0; j < len; ++j)
    printf("%f\n", res[j]);
  printf("Values End\n");
  return 0;
}
