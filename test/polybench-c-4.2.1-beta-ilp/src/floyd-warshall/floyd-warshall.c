#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "instrument.h"

#ifdef GLOBAL_ALLOC
#define PB_STATIC static
#else
#define PB_STATIC
#endif

#  ifdef MINI_DATASET
#   define N 60
#  endif

#  ifdef SMALL_DATASET
#   define N 180
#  endif

#  ifdef MEDIUM_DATASET
#   define N 500
#  endif

#  ifdef LARGE_DATASET
#   define N 2800
#  endif

#  ifdef EXTRALARGE_DATASET
#   define N 5600
#  endif
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.16lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)
#define POLYBENCH_DUMP_TARGET stdout

#define _PB_N N


int BENCH_MAIN(){


/* Variable declaration/allocation. */
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) path[N][N];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
/* Retrieve problem size. */
    int n = N;




    int i __attribute__((annotate("scalar(range(-60, 60))")));
    int j __attribute__((annotate("scalar(range(-60, 60))")));
    int k;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            path[i][j] = i*j%7+1;
            if ((i+j)%13 == 0 || (i+j)%7==0 || (i+j)%11 == 0)
                path[i][j] = 999;
        }

    for (k = 0; k < _PB_N; k++)
    {
        for(i = 0; i < _PB_N; i++)
            for (j = 0; j < _PB_N; j++) {
                DATA_TYPE __attribute__((annotate("scalar()"))) tmpa = path[i][k];
                DATA_TYPE __attribute__((annotate("scalar()"))) tmpb = path[k][j];
                int cond = path[i][j] < path[i][k] + path[k][j];
                if (cond) {
                    path[i][j] = path[i][j];
                } else {
                    path[i][j] = tmpa + tmpb;
                }
            }
    }


    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, path[i][j]);
        }

    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
