///TAFFO_TEST_ARGS -Xvra -propagate-all -Xvra -max-unroll=10
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#ifdef __APPLE__
#include <mach/mach_time.h>
#elif _WIN32
#include <windows.h>
#elif __linux__
#include <time.h>
#elif
#define NOBENCH
#endif


struct {
#ifdef __APPLE__
  uint64_t stime;
#elif _WIN32
  LONGLONG stime;
#elif __linux__
  struct timespec stime;
#endif
} timer_state;


void timerStart(void)
{
#ifdef __APPLE__
  timer_state.stime = mach_absolute_time();
#elif _WIN32
  QueryPerformanceCounter((LARGE_INTEGER*)&(timer_state.stime));
#elif __linux__
  clock_gettime(CLOCK_MONOTONIC_RAW, &(timer_state.stime));
#endif
}


uint64_t timerStop(void)
{
  uint64_t diff;

#ifdef __APPLE__
  uint64_t etime;
  mach_timebase_info_data_t info;

  etime = mach_absolute_time();
  mach_timebase_info(&info);
  diff = (etime - timer_state.stime) * info.numer / info.denom;
#elif _WIN32
  LONGLONG etime;
  LONGLONG freq;
  uint64_t tmp1, tmp2;

  QueryPerformanceCounter((LARGE_INTEGER*)&etime);
  QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
  diff = etime - timer_state.stime;
  tmp1 = (diff * 1000000) / freq;
  tmp2 = ((diff * 1000000) % freq) * 1000 / freq;
  diff = tmp1 * 1000 + tmp2;
#elif __linux__
  struct timespec etime;
  uint64_t t0, t1;

  clock_gettime(CLOCK_MONOTONIC_RAW, &etime);
  t0 = timer_state.stime.tv_nsec + timer_state.stime.tv_sec * 1000000000;
  t1 = etime.tv_nsec + etime.tv_sec * 1000000000;
  diff = t1 - t0;
#endif

  return diff;
}


uint64_t xorshift64star(void)
{
  static uint64_t x = UINT64_C(1970835257944453882);
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  return x * UINT64_C(2685821657736338717);
}


int randomInRange(int a, int b)
{
  int d = b - a;
  if (d <= 0)
    return a;
  uint64_t m = UINT64_MAX - UINT64_MAX % d;
  uint64_t n = xorshift64star();
  while (n > m)
    n = xorshift64star();
  return n % d + a;
}


#define N 0x40000
#define PREHEAT 10
#define TRIES 1000


#define gen_perform(op) {                                     \
  printf("operation: %s\n", #op);                             \
                                                              \
  uint64_t samples[TRIES];                                    \
                                                              \
  for (int t=0; t<TRIES; t++) {                               \
    timerStart();                                             \
    for (int i=0; i<N; i+=2) {                                \
      buf[i] = buf[i] op buf[i+1];                            \
    }                                                         \
    __attribute__((annotate("scalar(range(-3000, 3000))"))) float sync = 0.0;   \
    for (int i=0; i<N; i++)                                   \
      sync += buf[i];                                         \
    samples[t] = timerStop();                                 \
  }                                                           \
                                                              \
  uint64_t avg = 0;                                           \
  for (int t=PREHEAT; t<TRIES; t++) {                         \
    avg += samples[t];                                        \
  }                                                           \
  avg /= TRIES - PREHEAT;                                     \
  printf("avg time (ns): %" PRIu64 "\n", avg);                \
}


int main(int argc, char *argv[])
{
  __attribute__((annotate("scalar(range(-3000, 3000))"))) float buf[N*2];

  for (int i=0; i<N; i++) {
    buf[i] = (float)randomInRange(0, 0x100) / 32768.0;
  }
  gen_perform(+)
  
  for (int i=0; i<N; i++) {
    buf[i] = 1.0 + (float)randomInRange(0, (i+1) % 4) / 32768.0;
  }
  gen_perform(*)

  return 0;
}



