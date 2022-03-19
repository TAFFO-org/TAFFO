#include <stdio.h>
#include <stdlib.h>
#include "instrument.h"
#define DATA_TYPE double

#ifdef GLOBAL_ALLOC
#define PB_STATIC static
#else
#define PB_STATIC
#endif

#  ifdef MINI_DATASET
#   define M 28
#   define N 32
#  endif

#  ifdef SMALL_DATASET
#   define M 80
#   define N 100
#  endif

#  ifdef MEDIUM_DATASET
#   define M 240
#   define N 260
#  endif

#  ifdef LARGE_DATASET
#   define M 1200
#   define N 1400
#  endif

#  ifdef EXTRALARGE_DATASET
#   define M 2600
#   define N 3000
#  endif

#define _PB_M M
#define _PB_N N

int BENCH_MAIN(int argc, char** argv)
{


  PB_STATIC DATA_TYPE __attribute((annotate("scalar(range(-2097152, 2097151) final error(1e-100))"))) data[N][M];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(range(-2097152, 2097151) final error(1e-100))"))) cov[N][M];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(error(1e-100))"))) mean[M];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int n = N;
    int m = M;

    /* Variable declaration/allocation. */
    DATA_TYPE __attribute((annotate("scalar(range(-10000, 10000) final error(1e-100))"))) float_n;



    int __attribute((annotate("scalar(range(1, 260) final)"))) i;
    int __attribute((annotate("scalar(range(1, 260) final)"))) j;
    int __attribute((annotate("scalar(range(1, 260) final)"))) k;

    float_n = (DATA_TYPE)n;

    for (i = 0; i < N; i++)
        for (j = 0; j < M; j++)
            data[i][j] = ((DATA_TYPE) i*j) / M;






    for (j = 0; j < _PB_M; j++)
    {
        mean[j] = 0.0;
        for (i = 0; i < _PB_N; i++)
            mean[j] += data[i][j];
        mean[j] /= float_n;
    }



    for (i = 0; i < _PB_N; i++)
        for (j = 0; j < _PB_M; j++)
            data[i][j] -= mean[j];





    for (i = 0; i < _PB_M; i++)
        for (j = i; j < _PB_M; j++)
        {
            cov[i][j] = 0.0;
            for (k = 0; k < _PB_N; k++) {
                cov[i][j] += data[k][i] * data[k][j];
            }
            //cov[i][j] /= (float_n - 1.0);
            cov[i][j] /= (N - 1.0);
            cov[j][i] = cov[i][j];
        }



    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();

    for (i = 0; i < m; i++)
        for (j = 0; j < m; j++) {
            if ((i * m + j) % 20 == 0) fprintf (stdout, "\n");
            fprintf (stdout, "%.16lf ", cov[i][j]);
        }


    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
