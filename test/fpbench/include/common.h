#pragma once

#ifdef __x86_64__
#define CYCLES_START(cycles_high, cycles_low) \
  asm volatile("CPUID\n\t"                    \
               "RDTSC\n\t"                    \
               "mov %%edx, %0\n\t"            \
               "mov %%eax, %1\n\t"            \
               : "=r"(cycles_high), "=r"(cycles_low)::"%rax", "%rbx", "%rcx", "%rdx")
#else
#define CYCLES_START(cycles_high, cycles_low) \
  cycles_high = 0;                            \
  cycles_low = 0
#endif

#ifdef __x86_64__
#define CYCLES_END(cycles_high, cycles_low) \
  asm volatile("RDTSCP\n\t"                 \
               "mov %%edx, %0\n\t"          \
               "mov %%eax, %1\n\t"          \
               "CPUID\n\t"                  \
               : "=r"(cycles_high), "=r"(cycles_low)::"%rax", "%rbx", "%rcx", "%rdx")
#else
#define CYCLES_END(cycles_high, cycles_low) \
  cycles_high = 0;                          \
  cycles_low = 0
#endif
