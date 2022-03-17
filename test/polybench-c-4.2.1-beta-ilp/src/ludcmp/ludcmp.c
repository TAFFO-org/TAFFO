#include <stdio.h>
#include <stdlib.h>
#include "instrument.h"

#ifdef GLOBAL_ALLOC
#define PB_STATIC static
#else
#define PB_STATIC
#endif

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

#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.16lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)

#define POLYBENCH_DUMP_TARGET stdout

int BENCH_MAIN(){


/* Variable declaration/allocation. */
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-2, 2) final error(1e-100))"))) A[N][N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) b[N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) x[N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) y[N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar()"))) B[N][N];


    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int n = N;




    int i __attribute__((annotate("scalar(range(-400, 400) final)")));
    int j __attribute__((annotate("scalar(range(-400, 400) final)")));
    int k;
    DATA_TYPE __attribute__((annotate("scalar()"))) fn = (DATA_TYPE)n;

    for (i = 0; i < n; i++)
    {
        x[i] = 0;
        y[i] = 0;
        b[i] = (i+1)/(400.0)/2.0 + 4; //FIXED
    }

    for (i = 0; i < n; i++)
    {
        for (j = 0; j <= i; j++)
            A[i][j] = (DATA_TYPE)(-j % n) / 400.0 + 1; //FIXED
        for (j = i+1; j < n; j++) {
            A[i][j] = 0;
        }
        A[i][i] = 1;
    }

    /* Make the matrix positive semi-definite. */
    /* not necessary for LU, but using same code as cholesky */
    int r,s,t;
    for (r = 0; r < n; ++r)
        for (s = 0; s < n; ++s)
            ((B))[r][s] = 0;
    for (t = 0; t < n; ++t)
        for (r = 0; r < n; ++r)
            for (s = 0; s < n; ++s)
                ((B))[r][s] += A[r][t] * A[s][t];
    for (r = 0; r < n; ++r)
        for (s = 0; s < n; ++s)
            A[r][s] = ((B))[r][s];

    DATA_TYPE __attribute__((annotate("scalar(range(-2, 2) final error(1e-100))"))) w;

    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j <i; j++) {
            w = A[i][j];
            for (k = 0; k < j; k++) {
                w -= A[i][k] * A[k][j];
            }
            A[i][j] = w / (A[j][j]);
        }
        for (j = i; j < _PB_N; j++) {
            w = A[i][j];
            for (k = 0; k < i; k++) {
                w -= A[i][k] * A[k][j];
            }
            A[i][j] = w;
        }
    }

    for (i = 0; i < _PB_N; i++) {
        w = b[i];
        for (j = 0; j < i; j++)
            w -= A[i][j] * y[j];
        y[i] = w;
    }

    for (i = _PB_N-1; i >=0; i--) {
        w = y[i];
        for (j = i+1; j < _PB_N; j++)
            w -= A[i][j] * x[j];
        x[i] = w / (A[i][i]);
    }

    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();


    for (i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
    }

    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
