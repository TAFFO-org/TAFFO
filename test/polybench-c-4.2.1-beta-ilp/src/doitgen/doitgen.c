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
#   define NQ 8
#   define NR 10
#   define NP 12
#  endif

#  ifdef SMALL_DATASET
#   define NQ 20
#   define NR 25
#   define NP 30
#  endif

#  ifdef MEDIUM_DATASET
#   define NQ 40
#   define NR 50
#   define NP 60
#  endif

#  ifdef LARGE_DATASET
#   define NQ 140
#   define NR 150
#   define NP 160
#  endif

#  ifdef EXTRALARGE_DATASET
#   define NQ 220
#   define NR 250
#   define NP 270
#  endif

#   define _PB_NQ NQ
#   define _PB_NR NR
#   define _PB_NP NP


#define POLYBENCH_DUMP_TARGET stdout


int BENCH_MAIN(){


/* Variable declaration/allocation. */
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-32, 31) final error(1e-100))"))) A[NR][NQ][NP];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-32, 31) final error(1e-100))"))) sum[NP];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) C4[NP][NP];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int nr = NR;
    int nq = NQ;
    int np = NP;



    int i __attribute__((annotate("scalar(range(0, 60) final )")));
    int j __attribute__((annotate("scalar(range(0, 60) final )")));
    int k __attribute__((annotate("scalar(range(0, 60) final )")));

    int p, q, r, s;

    for (i = 0; i < nr; i++)
        for (j = 0; j < nq; j++)
            for (k = 0; k < np; k++)
                A[i][j][k] = (DATA_TYPE) ((i*j + k)%np) / np;
    for (i = 0; i < np; i++)
        for (j = 0; j < np; j++)
            C4[i][j] = (DATA_TYPE) (i*j % np) / np;

    for (r = 0; r < _PB_NR; r++)
        for (q = 0; q < _PB_NQ; q++)  {
            for (p = 0; p < _PB_NP; p++)  {
                sum[p] = SCALAR_VAL(0.0);
                for (s = 0; s < _PB_NP; s++)
                    sum[p] += A[r][q][s] * C4[s][p];
            }
            for (p = 0; p < _PB_NP; p++)
                A[r][q][p] = sum[p];
        }


    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();


    for (i = 0; i < nr; i++)
        for (j = 0; j < nq; j++)
            for (k = 0; k < np; k++) {
                if ((i*nq*np+j*np+k) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
                fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
            }

    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
