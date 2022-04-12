/*
 * Copyright (c) 2011-2014, Nicolas Limare <nicolas@limare.net>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

/**
 * @file timing.c
 * @brief timing and profiling tools
 *
 * @author Nicolas Limare <nicolas@limare.net>
 */

#ifndef USE_TIMING
#define USE_TIMING
#endif
#include "timing.h"

#if defined(STM32F407xx)
#include "stm32f407xx.h"
#include "system_stm32f4xx.h"
#elif defined(STM32F207xx)
#include "stm32f207xx.h"
#include "system_stm32f2xx.h"
#else
#error No STM32 architecture defined!
#endif


#ifndef TIMING_CPUCLOCK_NB
#define TIMING_CPUCLOCK_NB 16
#endif

/** CPU clock counter array, initialized to 0 (K&R2, p.86) */
unsigned long _timing_cpuclock_counter[TIMING_CPUCLOCK_NB];


void timing_cpuclock_init(void)
{
    DWT->CTRL &= ~DWT_CTRL_CYCCNTENA_Msk;
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}


unsigned long _timing_cpuclock(void)
{
    unsigned long dt = DWT->CYCCNT;
    return dt;
}


