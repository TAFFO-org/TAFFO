#ifndef INSTRUMENT_H
#define INSTRUMENT_H

#include <stdio.h>
#include <stdlib.h>
#include "stm32f207xx.h"
#include "system_stm32f2xx.h"

#define xstr(s) str(s)
#define str(s) #s


extern long unsigned dt;


static __attribute__((always_inline)) void TIMING_CPUCLOCK_START()
{
  DWT->CTRL &= ~DWT_CTRL_CYCCNTENA_Msk;
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  dt = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  __asm volatile("# LLVM-MCA-BEGIN " xstr(BENCH_MAIN));
}


static __attribute__((always_inline)) void TIMING_CPUCLOCK_TOGGLE()
{
  __asm volatile("# LLVM-MCA-END " xstr(BENCH_MAIN));
  dt = DWT->CYCCNT;
  DWT->CYCCNT = 0;
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
