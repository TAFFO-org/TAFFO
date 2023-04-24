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
