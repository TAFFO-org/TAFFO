#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

#define POLYBENCH_DUMP_TARGET stdout

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

int BENCH_MAIN(){


/* Variable declaration/allocation. */
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) r[N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-2, 2) final error(1e-100))"))) y[N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) z[N];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int n = N;




    int i __attribute__((annotate("scalar(range(-400, 400) final )")));
    int k  __attribute__((annotate("scalar(range(-400, 400) final )")));
    DATA_TYPE __attribute__((annotate("scalar(range(-2, 2) final error(1e-100))"))) alpha;
    DATA_TYPE __attribute__((annotate("scalar(range(-2, 2) final error(1e-100))"))) beta;
    DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) sum;

    for (i = 0; i < n; i++)
    {
        r[i] = (DATA_TYPE)(n+1-i) / (n*200.0) + 1.5;
    }






    /*y[0] = -r[0];
    beta = SCALAR_VAL(1.0);
    alpha = -r[0];*/
    //To fix a bug in TAFFO WTF
    k=0;
    y[k] = -r[k];
    beta = SCALAR_VAL(1.0);
    alpha = -r[k];

    for (k = 1; k < _PB_N; k++) {
        beta = (1-alpha*alpha)*beta;
        sum = SCALAR_VAL(0.0);
        for (i=0; i<k; i++) {
            sum += r[k-i-1]*y[i];
        }
        alpha = - (r[k] + sum)/beta;

        for (i=0; i<k; i++) {
            z[i] = y[i] + alpha*y[k-i-1];
        }
        for (i=0; i<k; i++) {
            y[i] = z[i];
        }
        y[k] = alpha;
    }
    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();
    for (i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
    }

    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
