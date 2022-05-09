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

#  ifdef MINI_DATASET
#   define NI 16
#   define NJ 18
#   define NK 20
#   define NL 22
#   define NM 24
#  endif

#  ifdef SMALL_DATASET
#   define NI 40
#   define NJ 50
#   define NK 60
#   define NL 70
#   define NM 80
#  endif

#  ifdef MEDIUM_DATASET
#   define NI 180
#   define NJ 190
#   define NK 200
#   define NL 210
#   define NM 220
#  endif

#  ifdef LARGE_DATASET
#   define NI 800
#   define NJ 900
#   define NK 1000
#   define NL 1100
#   define NM 1200
#  endif

#  ifdef EXTRALARGE_DATASET
#   define NI 1600
#   define NJ 1800
#   define NK 2000
#   define NL 2200
#   define NM 2400
#  endif

#   define _PB_NI NI
#   define _PB_NJ NJ
#   define _PB_NK NK
#   define _PB_NL NL
#   define _PB_NM NM

#define POLYBENCH_DUMP_TARGET stdout

int BENCH_MAIN(){

  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-16384, 16384) final error(1e-100))"))) E[NI][NJ];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) A[NI][NK];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) B[NK][NJ];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-16384, 16384) final error(1e-100))"))) F[NJ][NL];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) C[NJ][NM];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) D[NM][NL];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-16384, 16384) final error(1e-100))"))) G[NI][NL];


    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;
    int nm = NM;

    /* Variable declaration/allocation. */


    int i __attribute__((annotate("scalar(range(0, 220) final)")));
    int j __attribute__((annotate("scalar(range(0, 220) final)")));
    int k;

    for (i = 0; i < ni; i++)
        for (j = 0; j < nk; j++)
            A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / (5*ni);
    for (i = 0; i < nk; i++)
        for (j = 0; j < nj; j++)
            B[i][j] = (DATA_TYPE) ((i*(j+1)+2) % nj) / (5*nj);
    for (i = 0; i < nj; i++)
        for (j = 0; j < nm; j++)
            C[i][j] = (DATA_TYPE) (i*(j+3) % nl) / (5*nl);
    for (i = 0; i < nm; i++)
        for (j = 0; j < nl; j++)
            D[i][j] = (DATA_TYPE) ((i*(j+2)+2) % nk) / (5*nk);

    /* E := A*B */
    for (i = 0; i < _PB_NI; i++)
        for (j = 0; j < _PB_NJ; j++)
        {
            E[i][j] = SCALAR_VAL(0.0);
            for (k = 0; k < _PB_NK; ++k)
                E[i][j] += A[i][k] * B[k][j];
        }
    /* F := C*D */
    for (i = 0; i < _PB_NJ; i++)
        for (j = 0; j < _PB_NL; j++)
        {
            F[i][j] = SCALAR_VAL(0.0);
            for (k = 0; k < _PB_NM; ++k)
                F[i][j] += C[i][k] * D[k][j];
        }
    /* G := E*F */
    for (i = 0; i < _PB_NI; i++)
        for (j = 0; j < _PB_NL; j++)
        {
            G[i][j] = SCALAR_VAL(0.0);
            for (k = 0; k < _PB_NJ; ++k)
                G[i][j] += E[i][k] * F[k][j];
        }

    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();

    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++) {
            if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, G[i][j]);
        }

    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
