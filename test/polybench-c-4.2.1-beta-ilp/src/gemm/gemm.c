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
#   define NI 20
#   define NJ 25
#   define NK 30
#  endif

#  ifdef SMALL_DATASET
#   define NI 60
#   define NJ 70
#   define NK 80
#  endif

#  ifdef MEDIUM_DATASET
#   define NI 200
#   define NJ 220
#   define NK 240
#  endif

#  ifdef LARGE_DATASET
#   define NI 1000
#   define NJ 1100
#   define NK 1200
#  endif

#  ifdef EXTRALARGE_DATASET
#   define NI 2000
#   define NJ 2300
#   define NK 2600
#  endif

#   define _PB_NI NI
#   define _PB_NJ NJ
#   define _PB_NK NK
#define POLYBENCH_DUMP_TARGET stdout

int BENCH_MAIN(){


  PB_STATIC DATA_TYPE __attribute((annotate("scalar(range(-100, 100)  error(1e-100))"))) C[NI][NJ];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(range(-100, 100) error(1e-100))"))) A[NI][NK];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(range(-100, 100)  error(1e-100))"))) B[NK][NJ];


    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;
    int nk = NK;

    /* Variable declaration/allocation. */
    DATA_TYPE __attribute((annotate("scalar(error(1e-100))"))) alpha;
    DATA_TYPE __attribute((annotate("scalar(error(1e-100))"))) beta;

    int i, j, k;

    alpha = 1.5;
    beta = 1.2;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++)
            C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nk; j++)
            A[i][j] = (DATA_TYPE) (i*(j+1) % nk) / nk;
    for (i = 0; i < nk; i++)
        for (j = 0; j < nj; j++)
            B[i][j] = (DATA_TYPE) (i*(j+2) % nj) / nj;




    for (i = 0; i < _PB_NI; i++) {
        for (j = 0; j < _PB_NJ; j++)
            C[i][j] *= beta;
        for (k = 0; k < _PB_NK; k++) {
            for (j = 0; j < _PB_NJ; j++)
                C[i][j] += alpha * A[i][k] * B[k][j];
        }
    }


    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();

    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++) {
            if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
        }

    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
