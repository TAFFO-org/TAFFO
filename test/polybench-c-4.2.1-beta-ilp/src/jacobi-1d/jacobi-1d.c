#include <stdio.h>
#include <stdlib.h>
#include "instrument.h"

#ifdef GLOBAL_ALLOC
#define PB_STATIC static
#else
#define PB_STATIC
#endif

#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.16lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)
#  ifdef MINI_DATASET
#   define TSTEPS 20
#   define N 30
#  endif

#  ifdef SMALL_DATASET
#   define TSTEPS 40
#   define N 120
#  endif

#  ifdef MEDIUM_DATASET
#   define TSTEPS 100
#   define N 400
#  endif

#  ifdef LARGE_DATASET
#   define TSTEPS 500
#   define N 2000
#  endif

#  ifdef EXTRALARGE_DATASET
#   define TSTEPS 1000
#   define N 4000
#  endif
#define POLYBENCH_DUMP_TARGET stdout

#define _PB_N N
#define _PB_TSTEPS TSTEPS


int BENCH_MAIN(){


/* Variable declaration/allocation. */
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) A[N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) B[N];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int n = N;
    int tsteps = TSTEPS;




    int i __attribute__((annotate("scalar(range(-400,  400) final)")));
    int t __attribute__((annotate("scalar(range(-100,  100) final)")));

    for (i = 0; i < n; i++)
    {
        A[i] = ((DATA_TYPE) i+ 2) / n;
        B[i] = ((DATA_TYPE) i+ 3) / n;
    }

    for (t = 0; t < _PB_TSTEPS; t++)
    {
        for (i = 1; i < _PB_N - 1; i++)
            B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
        for (i = 1; i < _PB_N - 1; i++)
            A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
    }


    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();
    
    for (i = 0; i < n; i++)
    {
        if (i % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
        fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i]);
    }



    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
