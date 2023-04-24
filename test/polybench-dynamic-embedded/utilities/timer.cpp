#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

extern "C" {

/* Total LLC cache size. By default 32+MB.. */
#ifndef POLYBENCH_CACHE_SIZE_KB
# define POLYBENCH_CACHE_SIZE_KB 32770
#endif


/* Timer code (gettimeofday). */
double polybench_t_start, polybench_t_end;
/* Timer code (RDTSC). */
unsigned long long int polybench_c_start, polybench_c_end;

static double rtclock()
{
#if defined(POLYBENCH_TIME) || defined(POLYBENCH_GFLOPS)
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, NULL);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
#else
  return 0;
#endif
}

#ifdef POLYBENCH_CYCLE_ACCURATE_TIMER
static unsigned long long int rdtsc()
{
  unsigned long long int ret = 0;
  unsigned int cycles_lo;
  unsigned int cycles_hi;
  __asm__ volatile("RDTSC"
                   : "=a"(cycles_lo), "=d"(cycles_hi));
  ret = (unsigned long long int)cycles_hi << 32 | cycles_lo;

  return ret;
}
#endif

void polybench_flush_cache()
{
  int cs = POLYBENCH_CACHE_SIZE_KB * 1024 / sizeof(double);
  double *flush = (double *)calloc(cs, sizeof(double));
  int i;
  double tmp = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : tmp) private(i)
#endif
  for (i = 0; i < cs; i++)
    tmp += flush[i];
  //  assert (tmp <= 10.0);
  free(flush);
}

void polybench_prepare_instruments()
{
#ifndef POLYBENCH_NO_FLUSH_CACHE
  polybench_flush_cache();
#endif
#ifdef POLYBENCH_LINUX_FIFO_SCHEDULER
  polybench_linux_fifo_scheduler();
#endif
}

void polybench_timer_start()
{
  polybench_prepare_instruments();
#ifndef POLYBENCH_CYCLE_ACCURATE_TIMER
  polybench_t_start = rtclock();
#else
  polybench_c_start = TIMING_CPUCLOCK_START();
#endif
}

void polybench_timer_stop()
{
#ifndef POLYBENCH_CYCLE_ACCURATE_TIMER
  polybench_t_end = rtclock();
#else
  polybench_c_end = rdtsc();
#endif
#ifdef POLYBENCH_LINUX_FIFO_SCHEDULER
  polybench_linux_standard_scheduler();
#endif
}

void polybench_timer_print()
{
#ifdef POLYBENCH_GFLOPS
  if (polybench_program_total_flops == 0) {
    printf("[PolyBench][WARNING] Program flops not defined, use polybench_set_program_flops(value)\n");
    printf("%0.6lf\n", polybench_t_end - polybench_t_start);
  } else
    printf("%0.2lf\n",
           (polybench_program_total_flops /
            (double)(polybench_t_end - polybench_t_start)) /
               1000000000);
#else
#ifndef POLYBENCH_CYCLE_ACCURATE_TIMER
  printf("EXECUTION_TIME: %0.6f ms\n", (polybench_t_end - polybench_t_start) * 1000);
#else
  printf("%Ld\n", polybench_c_end - polybench_c_start);
#endif
#endif
}
}