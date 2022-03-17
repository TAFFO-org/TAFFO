#include <stdio.h>
#include <stdlib.h>
#include "instrument.h"

#ifdef GLOBAL_ALLOC
#define PB_STATIC static
#else
#define PB_STATIC
#endif

#define POLYBENCH_DUMP_TARGET stdout
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.16lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)

#  ifdef MINI_DATASET
#   define N 30
#  endif

#  ifdef SMALL_DATASET
#   define N 90
#  endif

#  ifdef MEDIUM_DATASET
#   define N 250
#  endif

#  ifdef LARGE_DATASET
#   define N 1300
#  endif

#  ifdef EXTRALARGE_DATASET
#   define N 2800
#  endif
#define _PB_N N


int BENCH_MAIN(){

  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) A[N][N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) B[N][N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-256, 255) final error(1e-100))"))) tmp[N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) x[N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-256, 255) final error(1e-100))"))) y[N];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) alpha;
    DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) beta;



    int i __attribute__((annotate("scalar(range(0,250) final)")));
    int j __attribute__((annotate("scalar(range(0,250) final)")));

    alpha = 1.5;
    beta = 1.2;
    for (i = 0; i < n; i++)
    {
        x[i] = (DATA_TYPE)( i % n) / n;
        for (j = 0; j < n; j++) {
            A[i][j] = (DATA_TYPE) ((i*j+1) % n) / n;
            B[i][j] = (DATA_TYPE) ((i*j+2) % n) / n;
        }
    }

    for (i = 0; i < _PB_N; i++)
    {
        tmp[i] = SCALAR_VAL(0.0);
        y[i] = SCALAR_VAL(0.0);
        for (j = 0; j < _PB_N; j++)
        {
            tmp[i] = A[i][j] * x[j] + tmp[i];
            y[i] = B[i][j] * x[j] + y[i];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
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
