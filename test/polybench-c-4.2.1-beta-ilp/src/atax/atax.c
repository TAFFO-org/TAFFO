#include <stdio.h>
#include <stdlib.h>
#include "instrument.h"
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.16lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)

#ifdef GLOBAL_ALLOC
#define PB_STATIC static
#else
#define PB_STATIC
#endif

#  ifdef MINI_DATASET
#   define M 38
#   define N 42
#  endif

#  ifdef SMALL_DATASET
#   define M 116
#   define N 124
#  endif

#  ifdef MEDIUM_DATASET
#   define M 390
#   define N 410
#  endif

#  ifdef LARGE_DATASET
#   define M 1900
#   define N 2100
#  endif

#  ifdef EXTRALARGE_DATASET
#   define M 1800
#   define N 2200
#  endif

#   define _PB_M M
#   define _PB_N N

#define POLYBENCH_DUMP_TARGET stdout

int BENCH_MAIN(){

  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) A[M][N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) x[N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-4096, 4096) final error(1e-100))"))) y[N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-4096, 4096) final error(1e-100))"))) tmp[M];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int m = M;
    int n = N;

    /* Variable declaration/allocation. */


    int i __attribute__((annotate("scalar(range(0, 410) final)")));
    int j __attribute__((annotate("scalar(range(0, 410) final)")));
    DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) fn;
    fn = (DATA_TYPE)n;

    for (i = 0; i < n; i++)
        x[i] = 1 + (i / fn);
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            A[i][j] = (DATA_TYPE) ((i+j) % n) / (5*m);

    for (i = 0; i < _PB_N; i++)
        y[i] = 0;
    for (i = 0; i < _PB_M; i++)
    {
        tmp[i] = SCALAR_VAL(0.0);
        for (j = 0; j < _PB_N; j++)
            tmp[i] = tmp[i] + A[i][j] * x[j];
        for (j = 0; j < _PB_N; j++)
            y[j] = y[j] + A[i][j] * tmp[i];
    }


    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();


    for (i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
    }

    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
