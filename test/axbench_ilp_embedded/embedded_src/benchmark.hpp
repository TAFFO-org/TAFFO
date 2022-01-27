#pragma once
#include <cstdint>
#include <cinttypes>
#include "stm32f207xx.h"
#include "system_stm32f2xx.h"


class AxBenchTimer {
public:
  AxBenchTimer()
  {
    reset();
  }
  
  
  void reset()
  {
    DWT->CTRL &= ~DWT_CTRL_CYCCNTENA_Msk;
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  }
  
  
  uint64_t cyclesSinceReset()
  {
    unsigned long dt = DWT->CYCCNT;
    return dt;
  }
};



