#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <time.h>
#include <stdint.h>


typedef struct {
  struct timespec stime;
} t_timer;

#define BENCH_ALWAYS_INLINE inline __attribute__((always_inline))


BENCH_ALWAYS_INLINE void timer_start(t_timer *state)
{
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &state->stime);
}

BENCH_ALWAYS_INLINE uint64_t timer_nsSinceStart(t_timer *state)
{
  struct timespec etime;
  uint64_t t0, t1;

  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &etime);
  t0 = state->stime.tv_nsec + state->stime.tv_sec * (uint64_t)1000000000;
  t1 = etime.tv_nsec + etime.tv_sec * (uint64_t)1000000000;
  return t1 - t0;
}

#define USE_REGISTER(x) asm volatile("" : "+r"(x) : :)

BENCH_ALWAYS_INLINE void use(void *value)
{
  asm volatile("" : : "r,m"(value) : "memory");
}


#endif
