#ifndef INSTRUMENT_H
#define INSTRUMENT_H

#include "stm32l010xb.h"
#include "stm32l0xx_hal.h"
#include "system_stm32l0xx.h"

#define xstr(s) str(s)
#define str(s) #s


extern long unsigned dt;
extern long unsigned last_reset_tick;


static __attribute__((always_inline)) void TIMING_CPUCLOCK_START()
{
  last_reset_tick = HAL_GetTick();
  dt = 0;
  __asm volatile("# LLVM-MCA-BEGIN " xstr(BENCH_MAIN));
}


static __attribute__((always_inline)) void TIMING_CPUCLOCK_TOGGLE()
{
  __asm volatile("# LLVM-MCA-END " xstr(BENCH_MAIN));
  dt = HAL_GetTick() - last_reset_tick;
}


static __attribute__((always_inline)) long unsigned TIMING_CPUCLOCK_S()
{
  return dt;
}


static __attribute__((always_inline)) void TIMING_CPUCLOCK_PRINT()
{
  printf("%ld clock cycles @ %ld Hz\n", dt, SystemCoreClock);
}


static __attribute__((always_inline)) void TAFFO_DUMPCONFIG()
{
  printf("****************** THIS IS BENCHMARK: %s\n", xstr(BENCH_MAIN));
}


#endif
