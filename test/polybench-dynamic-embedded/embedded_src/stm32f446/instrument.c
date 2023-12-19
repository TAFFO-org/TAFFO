#include "stm32f446xx.h"
#include "system_stm32f4xx.h"

#define xstr(s) str(s)
#define str(s) #s


long unsigned dt;
long unsigned last_reset_tick;

void reset()
{
  DWT->CTRL &= ~DWT_CTRL_CYCCNTENA_Msk;
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}


long unsigned cyclesSinceReset()
{
  return DWT->CYCCNT;
}

__attribute__((always_inline)) __attribute__((__visibility__("default"))) void polybench_timer_start()
{
  reset();
  __asm volatile("# LLVM-MCA-BEGIN " xstr(BENCH_MAIN));
}


__attribute__((always_inline)) __attribute__((__visibility__("default"))) void polybench_timer_stop()
{
  __asm volatile("# LLVM-MCA-END " xstr(BENCH_MAIN));
  dt = cyclesSinceReset();
}


__attribute__((always_inline)) __attribute__((__visibility__("default"))) void polybench_timer_print()
{
  printf("EXECUTION_TIME: %ld ms\n", dt);
}
