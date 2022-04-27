/*
 * Copyright (c) 2011-2014, Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under, at your option, the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version, or
 * the terms of the simplified BSD license.
 *
 * You should have received a copy of these licenses along this
 * program. If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

/**
 * @file timing.h
 * @brief timing and profiling macros
 *
 * These macros are only active if the USE_TIMING macro is
 * defined. Otherwise, they have no impact on the timed code.
 *
 * See timing_test.c for an example on how to use it.
 *
 * @author Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
 */

#ifndef _TIMING_H
#define _TIMING_H

#include <stdio.h>
#include <time.h>

/****************************************************************
 * CPU CLOCK TIMER
 ****************************************************************/

/**
 * @file timing.h
 *
 * CPU clock counters can be toggled on/off with
 * TIMING_CPUCLOCK_TOGGLE(N), with N between 0 and TIMING_CPUCLOCK_NB
 * - 1. Successive on/off toggle of the same counter, even from
 * different functions or different calls to the same function, will
 * add CPU clock time until the counter is reset with
 * TIMING_CPUCLOCK_RESET(N). The counter time can be read in seconds
 * (float) with TIMING_CPUCLOCK_S(N) and its raw value (long) with
 * TIMING_CPUCLOCK(N).
 *
 * Between the two toggles, the counter values are meaningless.
 * TIMING_CPUCLOCK_TOGGLE() must be called an even number of times to
 * make sense.
 *
 * These macros use different backend implementations (with different
 * precisions) from POSIX or libc standards depending on which is
 * available. They compute the the total CPU for the process, on all
 * CPUs: a process running on 8 CPUs for 2 seconds uses 16s of CPU
 * time. The CPU time is not affected directly by other programs
 * running on the machine, but there may be some side-effect because
 * of the ressource conflicts like CPU cache reload. These macros are
 * suitable when the measured time is around a second or more.
 *
 * Clock counters are not thread-safe. Your program will not crash
 * and die and burn if you use clock macros in a parallel program,
 * but the numbers may be wrong be if clock macros are called in
 * parallel. However, running some parallel instructions between two
 * clock macros is perfectly fine.
 *
 * @todo OpenMP-aware clock timing
 */

void timing_cpuclock_init(void);

/**
 * @brief portable cpuclock call
 */
unsigned long _timing_cpuclock(void);

/** CPU clock counter array, initialized to 0 */
extern unsigned long _timing_cpuclock_counter[];

/**
 * @brief reset a CPU clock counter
 */
#define TIMING_CPUCLOCK_RESET(N) { _timing_cpuclock_counter[N] = 0; }

/**
 * @brief toggle (start/stop) a CPU clock counter
 *
 * To measure the CPU clock time used by an instruction block, call
 * TIMING_CPUCLOCK_TOGGLE() before and after the block.
 *
 * The two successive substractions will increase the counter by the
 * difference of the successive CPU clocks. There is no overflow,
 * _timing_cpuclock_counter always stays between 0 and 2x the total
 * execution time.
 */
#define TIMING_CPUCLOCK_TOGGLE(N) {              \
        _timing_cpuclock_counter[N] = _timing_cpuclock() \
            - _timing_cpuclock_counter[N]; }

/**
 * @brief reset and toggle the CPU clock counter
 */
#define TIMING_CPUCLOCK_START(N) {              \
        TIMING_CPUCLOCK_RESET(N);               \
        TIMING_CPUCLOCK_TOGGLE(N); }

/**
 * @brief CPU clock time in seconds
 */
#define TIMING_CPUCLOCK_S(N) ((float) _timing_cpuclock_counter[N] / 1000000)

#endif                          /* !_TIMING_H */
