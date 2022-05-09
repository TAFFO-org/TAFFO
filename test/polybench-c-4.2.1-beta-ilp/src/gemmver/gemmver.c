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
#define _PB_N N

int BENCH_MAIN(){

  PB_STATIC DATA_TYPE __attribute((annotate("scalar(range(-2, 2) final error(1e-100))"))) A[N][N];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(error(1e-100))"))) u1[N];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(error(1e-100))"))) v1[N];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(error(1e-100))"))) u2[N];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(error(1e-100))"))) v2[N];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(range(-8000, 8000) final error(1e-100))"))) w[N];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(range(-30, 30) final error(1e-100))"))) x[N];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(error(1e-100))"))) y[N];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(error(1e-100))"))) z[N];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE __attribute((annotate("scalar(error(1e-100))"))) alpha;
    DATA_TYPE __attribute((annotate("scalar(error(1e-100))"))) beta;



    int i __attribute((annotate("scalar(range(0, 400) final )")));
    int j __attribute((annotate("scalar(range(0, 400) final )")));

    alpha = 1.5;
    beta = 1.2;

    DATA_TYPE __attribute((annotate("scalar()"))) fn = (DATA_TYPE)n;

    for (i = 0; i < n; i++)
    {
        u1[i] = i / fn;
        u2[i] = ((i+1)/fn)/2.0;
        v1[i] = ((i+1)/fn)/4.0;
        v2[i] = ((i+1)/fn)/6.0;
        y[i] = ((i+1)/fn)/8.0;
        z[i] = ((i+1)/fn)/9.0;
        x[i] = 0.0;
        w[i] = 0.0;
        for (j = 0; j < n; j++)
            A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }

    for (i = 0; i < _PB_N; i++)
        for (j = 0; j < _PB_N; j++)
            A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

    for (i = 0; i < _PB_N; i++)
        for (j = 0; j < _PB_N; j++)
            x[i] = x[i] + beta * A[j][i] * y[j];

    for (i = 0; i < _PB_N; i++)
        x[i] = x[i] + z[i];

    for (i = 0; i < _PB_N; i++)
        for (j = 0; j < _PB_N; j++)
            w[i] = w[i] +  alpha * A[i][j] * x[j];
    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();
    for (i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
    }

    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
