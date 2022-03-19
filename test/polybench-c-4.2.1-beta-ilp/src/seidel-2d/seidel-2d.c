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
#   define N 40
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

#   define _PB_TSTEPS TSTEPS
#   define _PB_N N

int BENCH_MAIN(){

  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) A[N][N];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
/* Retrieve problem size. */
    int n = N;
    int tsteps = TSTEPS;


    int i __attribute__((annotate("scalar(range(-400, 400) final)")));
    int j __attribute__((annotate("scalar(range(-400, 400) final)")));
    int t;


    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;

    for (t = 0; t <= _PB_TSTEPS - 1; t++)
        for (i = 1; i<= _PB_N - 2; i++)
            for (j = 1; j <= _PB_N - 2; j++) {
                DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) term1 = A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
                                                                        + A[i][j-1];
                DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) term2 = A[i][j] + A[i][j+1]
                                                                        + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1];
                DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) sum = term1 + term2;
                DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) div = sum / SCALAR_VAL(9.0);
                A[i][j] = div;
            }
    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
            fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
        }

    return 0;

}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
