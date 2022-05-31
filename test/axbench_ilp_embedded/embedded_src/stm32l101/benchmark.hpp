#pragma once
#include <cstdint>
#include <cinttypes>
#include "stm32l010xb.h"
#include "stm32l0xx_hal.h"
#include "system_stm32l0xx.h"


class AxBenchTimer {
public:
  AxBenchTimer()
  {
    reset();
  }
  
  
  void reset()
  {
    last_reset_tick = HAL_GetTick();
  }
  
  
  uint32_t cyclesSinceReset()
  {
    uint32_t dt = HAL_GetTick() - last_reset_tick;
    return dt;
  }

private:
  unsigned long last_reset_tick;

};



