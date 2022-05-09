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

#define POLYBENCH_DUMP_TARGET stdout


int BENCH_MAIN(){

/* Variable declaration/allocation. */
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-10, 10) final error(1e-100))"))) A[N][N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) B[N][N];


    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
/* Retrieve problem size. */
    int n = N;



    int i __attribute__((annotate("scalar(range(-400, 400) final)")));
    int j __attribute__((annotate("scalar(range(-400, 400) final)")));
    int k;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j <= i; j++)
            A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
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


    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j <i; j++) {
            for (k = 0; k < j; k++) {
                DATA_TYPE __attribute__((annotate("scalar(range(-10, 10) final  error(1e-100))"))) tmp = A[i][k] * A[k][j];
                A[i][j] -= tmp;
            }
            A[i][j] /= A[j][j];
        }
        for (j = i; j < _PB_N; j++) {
            for (k = 0; k < i; k++) {
                DATA_TYPE __attribute__((annotate("scalar(range(-10, 10) final error(1e-100))"))) tmp = A[i][k] * A[k][j];
                A[i][j] -= tmp;
            }
        }
    }
    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
        }

    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
