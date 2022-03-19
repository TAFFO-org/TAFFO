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
#define DATA_TYPE double


int BENCH_MAIN(){


  PB_STATIC DATA_TYPE __attribute((annotate("scalar(range(-50000, 50000) final error(1e-100))"))) mean[M];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(range(-10, 10) error(1e-100) final)"))) data[N][M];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(range(0, 5) error(1e-100) final)"))) corr[M][M];
  PB_STATIC DATA_TYPE __attribute((annotate("scalar(range(-4096,4096) error(1e-1) final)"))) stddev[M];


    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();

    int __attribute((annotate("scalar(range(0, 240) final )"))) i;
    int __attribute((annotate("scalar(range(0, 260) final )"))) j;
    int __attribute((annotate("scalar(range(0, 260) final )"))) k;


    DATA_TYPE __attribute((annotate("scalar(range(1, 3000) error(1e-100))"))) float_n;

    float_n = (DATA_TYPE)N;

    for (i = 0; i < N; i++)
        for (j = 0; j < M; j++)
            data[i][j] = ((DATA_TYPE)(i*j)/M + i)/N;

    //Print
    /*double min = data[0][0];
    double max = data[0][0];
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            if(data[i][j] > max) max = data[i][j];
            if(data[i][j] < min) min = data[i][j];
        }
    }
    printf("min: %f, max: %f\n", min, max);*/



    ///KERNEL
    DATA_TYPE __attribute((annotate("scalar()"))) eps = (0.1);
    for (j = 0; j < M; j++)
    {
        mean[j] = (0.0);
        for (i = 0; i < N; i++)
            mean[j] += data[i][j];
        mean[j] /= float_n;
    }


    for (j = 0; j < M; j++)
    {
        stddev[j] = (0.0);
        for (i = 0; i < N; i++)
            stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
        stddev[j] /= float_n;
        stddev[j] = sqrt(stddev[j]);
        /* The following in an inelegant but usual way to handle
           near-zero std. dev. values, which below would cause a zero-
           divide. */
        stddev[j] = stddev[j] <= eps ? (1.0) : stddev[j];
    }

    /* Center and reduce the column vectors. */
    for (i = 0; i < N; i++)
        for (j = 0; j < M; j++)
        {
            data[i][j] -= mean[j];
            data[i][j] /= sqrt(float_n) * stddev[j];
        }

    /*//Print
    min = data[0][0];
    max = data[0][0];
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            if(data[i][j] > max) max = data[i][j];
            if(data[i][j] < min) min = data[i][j];
        }
    }
    printf("min: %f, max: %f\n", min, max);*/

    /* Calculate the m * m correlation matrix. */
    for (i = 0; i < M-1; i++)
    {
        corr[i][i] = (1.0);
        for (j = i+1; j < M; j++)
        {
            corr[i][j] = (0.0);
            for (k = 0; k < N; k++)
                corr[i][j] += (data[k][i] * data[k][j]);
            corr[j][i] = corr[i][j];
        }
    }
    corr[M-1][M-1] = (1.0);

    /*//Print
    min = corr[0][0];
    max = corr[0][0];
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            if(corr[i][j] > max) max = corr[i][j];
            if(corr[i][j] < min) min = corr[i][j];
        }
    }
    printf("CORR min: %f, max: %f\n", min, max);*/

    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();

    //Print
    //printf("\n\nCORR:\n");
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            printf("%.16lf ", corr[i][j]);
        }
        printf("\n");
    }

    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
