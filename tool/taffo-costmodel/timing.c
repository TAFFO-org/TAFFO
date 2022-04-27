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

#include "timing.h"

/*
 * OS DETECTION
 */

#if (defined(_WIN32) || defined(__WIN32__) \
     || defined(__TOS_WIN__) || defined(__WINDOWS__))
/* from http://sourceforge.net/p/predef/wiki/OperatingSystems/ */
#define TIMING_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#elif (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))
/* from http://sourceforge.net/p/predef/wiki/Standards/ */
#include <unistd.h>
#if (defined(_POSIX_VERSION) && (_POSIX_VERSION >= 200112L))
#define TIMING_POSIX
#include <sys/time.h>
#endif                          /* POSIX test */
#endif                          /* Windows/Unix test */

/*
 * ARCHITECTURE DETECTION
 */

#if (defined(__amd64__) || defined(__amd64) || defined(_M_X64))
/* from http://sourceforge.net/p/predef/wiki/Architectures/ */
#define TIMING_AMD64

#elif (defined(__i386__) || defined(__i386) || defined(_M_IX86) \
       || defined(__X86__) || defined(_X86_) || defined(__I86__))
/* from http://predef.sourceforge.net/prearch.html#sec6 */
#define TIMING_I386
#endif


/****************************************************************
 * CPU CLOCK TIMER
 ****************************************************************/

/** number of CPU clock counters */
#ifndef TIMING_CPUCLOCK_NB
#define TIMING_CPUCLOCK_NB 16
#endif

/** CPU clock counter array, initialized to 0 (K&R2, p.86) */
unsigned long _timing_cpuclock_counter[TIMING_CPUCLOCK_NB];

void timing_cpuclock_init(void)
{
  return;
}

/**
 * @brief portable cpuclock call
 */
#if defined(TIMING_POSIX)
/* use POSIX clock_gettime (reduced to microsecond precision) */
unsigned long _timing_cpuclock()
{
    struct timespec tmp;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tmp);
    return (unsigned long)(tmp.tv_sec * 1000000 + tmp.tv_nsec / 1000);
}
#else
unsigned long _timing_cpuclock()
/* fall back to libc clock() (1/100s to 1/10000s precision) */
{
    return (unsigned long)(clock() * 1000000 / CLOCKS_PER_SEC);
}
#endif
