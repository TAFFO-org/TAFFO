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
#   define M 20
#   define N 30
#  endif

#  ifdef SMALL_DATASET
#   define M 60
#   define N 80
#  endif

#  ifdef MEDIUM_DATASET
#   define M 200
#   define N 240
#  endif

#  ifdef LARGE_DATASET
#   define M 1000
#   define N 1200
#  endif

#  ifdef EXTRALARGE_DATASET
#   define M 2000
#   define N 2600
#  endif

#   define _PB_M M
#   define _PB_N N


int BENCH_MAIN(){

  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) A[M][M];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-256, 255) final error(1e-100))"))) B[M][N];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int m = M;
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) alpha;

    int i __attribute__((annotate("scalar(range(0, 200) final)")));
    int j __attribute__((annotate("scalar(range(0, 240) final)")));
    int k;

    alpha = 1.5;
    for (i = 0; i < m; i++) {
        for (j = 0; j < i; j++) {
            A[i][j] = (DATA_TYPE)((i+j) % m)/m;
        }
        A[i][i] = 1.0;
        for (j = 0; j < n; j++) {
            B[i][j] = (DATA_TYPE)((n+(i-j)) % n)/n;
        }
    }

    for (i = 0; i < _PB_M; i++)
        for (j = 0; j < _PB_N; j++) {
            for (k = i+1; k < _PB_M; k++)
                B[i][j] += A[k][i] * B[k][j];
            B[i][j] = alpha * B[i][j];
        }

    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, B[i][j]);
        }

    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
