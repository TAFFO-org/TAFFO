#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "instrument.h"

#ifdef GLOBAL_ALLOC
#define PB_STATIC static
#else
#define PB_STATIC
#endif

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

#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.16lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)


int BENCH_MAIN(){

/* Variable declaration/allocation. */
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-1000, 1000) final error(1e-100))"))) A[M][N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-1000, 1000) final error(1e-100))"))) R[N][N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-1000, 1000) final error(1e-100))"))) Q[M][N];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int m = M;
    int n = N;



    int i __attribute__((annotate("scalar(range(-30, 30) final)")));
    int j __attribute__((annotate("scalar(range(-30, 30) final)")));
    int k;

    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            A[i][j] = (((DATA_TYPE) ((i*j) % m) / m )*100) + 10;
            Q[i][j] = 0.0;
        }
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            R[i][j] = 0.0;




    DATA_TYPE __attribute__((annotate("scalar(range(-1000, 1000) final error(1e-100))"))) nrm;
    for (k = 0; k < _PB_N; k++)
    {
        nrm = SCALAR_VAL(0.0);
        for (i = 0; i < _PB_M; i++)
            nrm += A[i][k] * A[i][k];
        R[k][k] = SQRT_FUN(nrm);
        for (i = 0; i < _PB_M; i++)
            if (R[k][k] != 0)
                Q[i][k] = A[i][k] / R[k][k];
            else
                Q[i][k] = 0.0;
        for (j = k + 1; j < _PB_N; j++)
        {
            R[k][j] = SCALAR_VAL(0.0);
            for (i = 0; i < _PB_M; i++)
                R[k][j] += Q[i][k] * A[i][j];
            for (i = 0; i < _PB_M; i++)
                A[i][j] = A[i][j] - Q[i][k] * R[k][j];
        }
    }

    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, R[i][j]);
        }


    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, Q[i][j]);
        }


    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
