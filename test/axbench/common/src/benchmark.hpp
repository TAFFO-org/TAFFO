#include <cstdint>
#include <cinttypes>
#ifdef __APPLE__
#include <mach/mach_time.h>
#elif _WIN32
#include <windows.h>
#elif __linux__
#include <time.h>
#endif

#pragma once


class AxBenchTimer {
  #ifdef __APPLE__
    uint64_t stime;
  #elif _WIN32
    LONGLONG stime;
  #elif __linux__
    struct timespec stime;
  #endif
  
public:
  AxBenchTimer() 
  {
    reset();
  }
  
  
  void reset()
  {
    #ifdef __APPLE__
      stime = mach_absolute_time();
    #elif _WIN32
      QueryPerformanceCounter((LARGE_INTEGER*)&stime);
    #elif __linux__
      clock_gettime(CLOCK_MONOTONIC_RAW, &stime);
    #endif
  }
  
  
  uint64_t nanosecondsSinceInit()
  {
    uint64_t diff;
  
    #ifdef __APPLE__
      uint64_t etime;
      mach_timebase_info_data_t info;
  
      etime = mach_absolute_time();
      mach_timebase_info(&info);
      diff = (etime - stime) * info.numer / info.denom;
    #elif _WIN32
      LONGLONG etime;
      LONGLONG freq;
      uint64_t tmp1, tmp2;
  
      QueryPerformanceCounter((LARGE_INTEGER*)&etime);
      QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
      diff = etime - stime;
      tmp1 = (diff * 1000000) / freq;
      tmp2 = ((diff * 1000000) % freq) * 1000 / freq;
      diff = tmp1 * 1000 + tmp2;
    #elif __linux__
      struct timespec etime;
      uint64_t t0, t1;

      clock_gettime(CLOCK_MONOTONIC_RAW, &etime);
      t0 = stime.tv_nsec + stime.tv_sec * 1000000000;
      t1 = etime.tv_nsec + etime.tv_sec * 1000000000;
      diff = t1 - t0;
    #else
      #error unsupported platform
    #endif

    return diff;
  }
};



