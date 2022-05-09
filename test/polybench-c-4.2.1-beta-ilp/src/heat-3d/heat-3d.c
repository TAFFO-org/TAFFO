#include <stdio.h>
#include <stdlib.h>
#include "instrument.h"

#ifdef GLOBAL_ALLOC
#define PB_STATIC static
#else
#define PB_STATIC
#endif

#define DATA_TYPE double
#  ifdef MINI_DATASET
#   define TSTEPS 20
#   define N 10
#  endif

#  ifdef SMALL_DATASET
#   define TSTEPS 40
#   define N 20
#  endif

#  ifdef MEDIUM_DATASET
#   define TSTEPS 100
#   define N 40
#  endif

#  ifdef LARGE_DATASET
#   define TSTEPS 500
#   define N 120
#  endif

#  ifdef EXTRALARGE_DATASET
#   define TSTEPS 1000
#   define N 200
#  endif
#define _PB_N N

#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.16lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)

#define POLYBENCH_DUMP_TARGET stdout


int BENCH_MAIN(){

/* Variable declaration/allocation. */
	PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-50, 50) error(1e-100))"))) A[N][N][N];
	PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-50, 50) error(1e-100))"))) B[N][N][N];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    int n = N;
    int tsteps = TSTEPS;



    int t __attribute__((annotate("scalar(range(0, 80) final error(1e-100))")));
    int i __attribute__((annotate("scalar(range(0, 80) final error(1e-100))")));
    int j __attribute__((annotate("scalar(range(0, 80) final error(1e-100))")));
    int k __attribute__((annotate("scalar(range(0, 80) final error(1e-100))")));

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (k = 0; k < n; k++)
                A[i][j][k] = B[i][j][k] = (DATA_TYPE) (i + j + (n-k))* 10 / (n);


    for (t = 1; t <= TSTEPS; t++) {
        for (i = 1; i < _PB_N-1; i++) {
            for (j = 1; j < _PB_N-1; j++) {
                for (k = 1; k < _PB_N-1; k++) {
                    B[i][j][k] =   SCALAR_VAL(0.125) * (A[i+1][j][k] - SCALAR_VAL(2.0) * A[i][j][k] + A[i-1][j][k])
                                   + SCALAR_VAL(0.125) * (A[i][j+1][k] - SCALAR_VAL(2.0) * A[i][j][k] + A[i][j-1][k])
                                   + SCALAR_VAL(0.125) * (A[i][j][k+1] - SCALAR_VAL(2.0) * A[i][j][k] + A[i][j][k-1])
                                   + A[i][j][k];
                }
            }
        }
        for (i = 1; i < _PB_N-1; i++) {
            for (j = 1; j < _PB_N-1; j++) {
                for (k = 1; k < _PB_N-1; k++) {
                    A[i][j][k] =   SCALAR_VAL(0.125) * (B[i+1][j][k] - SCALAR_VAL(2.0) * B[i][j][k] + B[i-1][j][k])
                                   + SCALAR_VAL(0.125) * (B[i][j+1][k] - SCALAR_VAL(2.0) * B[i][j][k] + B[i][j-1][k])
                                   + SCALAR_VAL(0.125) * (B[i][j][k+1] - SCALAR_VAL(2.0) * B[i][j][k] + B[i][j][k-1])
                                   + B[i][j][k];
                }
            }
        }
    }

    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();


    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (k = 0; k < n; k++) {
                if ((i * n * n + j * n + k) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
                fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
            }


    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
