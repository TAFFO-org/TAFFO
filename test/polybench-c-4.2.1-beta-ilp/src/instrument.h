#ifndef INSTRUMENT_H
#define INSTRUMENT_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BENCH_MAIN
#define BENCH_MAIN main
#endif

#define xstr(s) str(s)
#define str(s) #s

long unsigned time_that_takes;

long unsigned gettime() __attribute__((always_inline)) {
    struct timespec tmp;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tmp);
    return (unsigned long)(tmp.tv_sec * 1000000 + tmp.tv_nsec / 1000);
}

#define TIMING_CPUCLOCK_START() \
  do { \
    time_that_takes = gettime(); \
    __asm volatile("# LLVM-MCA-BEGIN " xstr(BENCH_MAIN)); \
  } while (0)


#define TIMING_CPUCLOCK_TOGGLE() \
  do { \
    __asm volatile("# LLVM-MCA-END " xstr(BENCH_MAIN)); \
    time_that_takes = gettime() - time_that_takes; \
  } while (0)


#define TIMING_CPUCLOCK_S() time_that_takes


#define TIMING_CPUCLOCK_PRINT() \
  do { \
    fprintf(stderr, "%ld", time_that_takes); \
  } while (0)


#define TAFFO_DUMPCONFIG() do; while (0)


#endif
