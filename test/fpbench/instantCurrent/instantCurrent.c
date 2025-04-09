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
double msin(double x)
{
  return x - ((x * x * x) / 6.0f);
}

double cos(double x)
{
  return 1.0f - (x * x * 0.25f);
}

#else
#include <math.h>
#endif


float ex0(float t, float resistance, float frequency, float inductance,
          float maxVoltage)
{
  float pi = 3.14159265359f;
  float impedance_re = resistance;
  float __attribute__((annotate("scalar(range(-1, 4))"))) impedance_im = ((2.0f * pi) * frequency) * inductance;
  float __attribute__((annotate("scalar(range(-1, 2503))"))) denom = (impedance_re * impedance_re) + (impedance_im * impedance_im);
  float re_tmp = (maxVoltage * impedance_re) ;
  float im_tmp = (maxVoltage * impedance_im) ;
   im_tmp = -im_tmp;

  float  current_re = re_tmp / denom;
  float  current_im = im_tmp / denom;
  float __attribute__((annotate("scalar(range(-1, 11))"))) maxCurrent =
      sqrt(((current_re * current_re) + (current_im * current_im)));


  float __attribute__((annotate("scalar(range(-1, 4))"))) theta = atan((current_im / current_re));
  float cos_1 = (2.0f * pi);
  float cos_2 = (cos_1 * frequency);
  float cos_3 = ( cos_2 * t);
  float __attribute__((annotate("scalar(range(-8478652928, 13320812))"))) cos_4 = cos_3 ;
  float __attribute__((annotate("scalar(range(-8478652928, 13320812))"))) cos_5 = cos(cos_4);
  float __attribute__((annotate("scalar(range(-8478652928, 13320812))")))  mmaxCurrent = maxCurrent;
  float __attribute__((annotate("scalar(range(-8478652928, 13320812))")))  tmp = mmaxCurrent * cos_5;
  return tmp;
}

int main()
{
  static const int len = sizeof(arr) / sizeof(arr[0]) / 5;
  float __attribute__((annotate("target('main') scalar(range(-2, 300))")))
  t[len];
  float __attribute__((annotate("scalar(range(-10, 50))"))) resistance[len];
  float __attribute__((annotate("scalar(range(-10, 100))"))) frequency[len];
  float __attribute__((annotate("scalar(range(-2, 2))")))
  inductance[len];
  float __attribute__((annotate("scalar(range(-2, 12))"))) maxVoltage[len];
  float __attribute__((annotate("scalar(range(-8478652928, 13320812))"))) res[len];
  for (int i = 0; i < len; ++i) {

    t[i] = arr[i * 5];
    resistance[i] = arr[i * 5 + 1];
    frequency[i] = arr[i * 5 + 2];
    inductance[i] = arr[i * 5 + 3];
    maxVoltage[i] = arr[i * 5 + 4];
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
      res[j] =
          ex0(t[j], resistance[j], frequency[j], inductance[j], maxVoltage[j]);
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