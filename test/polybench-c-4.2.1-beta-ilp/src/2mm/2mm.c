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

/*#   define NI 180
#   define NJ 190
#   define NK 210
#   define NL 220
*/

#  ifdef MINI_DATASET
#   define NI 16
#   define NJ 18
#   define NK 22
#   define NL 24
#  endif

#  ifdef SMALL_DATASET
#   define NI 40
#   define NJ 50
#   define NK 70
#   define NL 80
#  endif

#  ifdef MEDIUM_DATASET
#   define NI 180
#   define NJ 190
#   define NK 210
#   define NL 220
#  endif

#  ifdef LARGE_DATASET
#   define NI 800
#   define NJ 900
#   define NK 1100
#   define NL 1200
#  endif

#  ifdef EXTRALARGE_DATASET
#   define NI 1600
#   define NJ 1800
#   define NK 2200
#   define NL 2400
#  endif

#   define _PB_NI NI
#   define _PB_NJ NJ
#   define _PB_NK NK
#   define _PB_NL NL

#define POLYBENCH_DUMP_TARGET stdout

int BENCH_MAIN(){
    PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-16384, 16384) final error(1e-100))"))) tmp[NI][NJ];
    PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) A[NI][NK];
    PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) B[NK][NJ];
    PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) C[NJ][NL];
    PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-16384, 16384) final error(1e-100))"))) D[NI][NL];
    
    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;

    /* Variable declaration/allocation. */
    DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) alpha;
    DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) beta;

    int i __attribute__((annotate("scalar(range(0, 210) final)")));
    int j __attribute__((annotate("scalar(range(0, 220) final)")));
    int k;

    alpha = 1.5;
    beta = 1.2;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nk; j++)
            A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
    for (i = 0; i < nk; i++)
        for (j = 0; j < nj; j++)
            B[i][j] = (DATA_TYPE) (i*(j+1) % nj) / nj;
    for (i = 0; i < nj; i++)
        for (j = 0; j < nl; j++)
            C[i][j] = (DATA_TYPE) ((i*(j+3)+1) % nl) / nl;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++)
            D[i][j] = (DATA_TYPE) (i*(j+2) % nk) / nk;

    for (i = 0; i < _PB_NI; i++)
        for (j = 0; j < _PB_NJ; j++)
        {
            tmp[i][j] = SCALAR_VAL(0.0);
            for (k = 0; k < _PB_NK; ++k)
                tmp[i][j] += alpha * A[i][k] * B[k][j];
        }
    for (i = 0; i < _PB_NI; i++)
        for (j = 0; j < _PB_NL; j++)
        {
            D[i][j] *= beta;
            for (k = 0; k < _PB_NJ; ++k)
                D[i][j] += tmp[i][k] * C[k][j];
        }

    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++) {
            if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D[i][j]);
        }


    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
