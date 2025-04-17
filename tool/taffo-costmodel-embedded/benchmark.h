#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stdint.h>
#include <time.h>

#if defined(STM32F407xx)
#include "stm32f407xx.h"
#include "system_stm32f4xx.h"
#elif defined(STM32F207xx)
#include "stm32f207xx.h"
#include "system_stm32f2xx.h"
#else
#error No STM32 architecture defined!
#endif

typedef struct {
  int dummy;
} t_timer;

#define BENCH_ALWAYS_INLINE inline __attribute__((always_inline))

BENCH_ALWAYS_INLINE void timer_start(t_timer* state) {
  DWT->CTRL &= ~DWT_CTRL_CYCCNTENA_Msk;
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

BENCH_ALWAYS_INLINE uint64_t timer_nsSinceStart(t_timer* state) {
  uint64_t dt = DWT->CYCCNT;
  return dt;
}

#define USE_REGISTER(x) asm volatile("" : "+r"(x) : :)

BENCH_ALWAYS_INLINE void use(void* value) { asm volatile("" : : "r,m"(value) : "memory"); }

#endif
