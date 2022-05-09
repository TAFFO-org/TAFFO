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

#define POLYBENCH_DUMP_TARGET stdout
#  ifdef MINI_DATASET
#   define N 40
#  endif

#  ifdef SMALL_DATASET
#   define N 120
#  endif

#  ifdef MEDIUM_DATASET
#   define N 400
#  endif

#  ifdef LARGE_DATASET
#   define N 2000
#  endif

#  ifdef EXTRALARGE_DATASET
#   define N 4000
#  endif
#   define _PB_N N



int BENCH_MAIN(){

/* Variable declaration/allocation. */
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) L[N][N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-1, 1) final error(1e-100))"))) x[N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) b[N];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int n = N;




    int i __attribute__((annotate("scalar(range(-8000, 8000) final)")));
    int j __attribute__((annotate("scalar(range(-8000, 8000) final)")));

    for (i = 0; i < n; i++)
    {
        x[i] = - 999;
        b[i] =  i ;
        for (j = 0; j <= i; j++)
            L[i][j] = (DATA_TYPE) (i+n-j+1)*2/n;
    }

    for (i = 0; i < _PB_N; i++)
    {
        x[i] = b[i];
        for (j = 0; j <i; j++)
            x[i] -= L[i][j] * x[j];
        x[i] = x[i] / L[i][i];
    }
    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();
    for (i = 0; i < n; i++) {
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    }

    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
