#include <stdlib.h>
#include "stm32l010xb.h"
#include "stm32l0xx_hal.h"
#include "system_stm32l0xx.h"

#define xstr(s) str(s)
#define str(s) #s


long unsigned dt;
long unsigned last_reset_tick;

__attribute__((always_inline)) __attribute__((__visibility__("default"))) void polybench_timer_start()
{
  last_reset_tick = HAL_GetTick();
  dt = 0;
  __asm volatile("# LLVM-MCA-BEGIN " xstr(BENCH_MAIN));
}


__attribute__((always_inline)) __attribute__((__visibility__("default"))) void polybench_timer_stop()
{
  __asm volatile("# LLVM-MCA-END " xstr(BENCH_MAIN));
  dt = HAL_GetTick() - last_reset_tick;
}


__attribute__((always_inline)) __attribute__((__visibility__("default"))) void polybench_timer_print()
{
  printf("EXECUTION_TIME: %ld ms\n", dt);
}

float __attribute__((__visibility__("default"))) RandomNumber(float Min, float Max)
{
  float r = (float)rand() / (float)RAND_MAX;
  float result = r  * (Max - Min) + Min;
  return result;
}

//void __attribute__((__visibility__("default"))) randomize_scalar(float *val, float amplitude) {
//  *val += *val * RandomNumber(-amplitude, amplitude);
//}
//
//void __attribute__((__visibility__("default"))) randomize_1d(int n, float val[n], float amplitude) {
//  for (int i = 0; i < n; i++) {
//    val[i] += val[i] * RandomNumber(-amplitude, amplitude);
//  }
//}
//
//void __attribute__((__visibility__("default"))) randomize_2d(int n, int m, float val[n][m], float amplitude) {
//  for (int i = 0; i < n; i++) {
//    for (int j = 0; j < m; j++) {
//      val[i][j] += val[i][j] * RandomNumber(-amplitude, amplitude);
//    }
//  }
//}
//
//void __attribute__((__visibility__("default"))) randomize_3d(int n, int m, int p, float val[n][m][p], float amplitude) {
//  for (int i = 0; i < n; i++) {
//    for (int j = 0; j < m; j++) {
//      for (int k = 0; k < p; k++) {
//        val[i][j][k] += val[i][j][k] * RandomNumber(-amplitude, amplitude);
//      }
//    }
//  }
//}
