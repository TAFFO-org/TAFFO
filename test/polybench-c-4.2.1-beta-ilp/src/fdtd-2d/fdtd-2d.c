#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "instrument.h"

#define DATA_TYPE double

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
#   define TMAX 20
#   define NX 20
#   define NY 30
#  endif

#  ifdef SMALL_DATASET
#   define TMAX 40
#   define NX 60
#   define NY 80
#  endif

#  ifdef MEDIUM_DATASET
#   define TMAX 100
#   define NX 200
#   define NY 240
#  endif

#  ifdef LARGE_DATASET
#   define TMAX 500
#   define NX 1000
#   define NY 1200
#  endif

#  ifdef EXTRALARGE_DATASET
#   define TMAX 1000
#   define NX 2000
#   define NY 2600
#  endif

#   define _PB_TMAX TMAX
#   define _PB_NX NX
#   define _PB_NY NY

#define POLYBENCH_DUMP_TARGET stdout


int BENCH_MAIN(){


/* Variable declaration/allocation. */
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-200, 200) final error(1e-100))"))) ex[NX][NY];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-200, 200) final error(1e-100))"))) ey[NX][NY];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-200, 200) final error(1e-100))"))) hz[NX][NY];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) _fict_[TMAX];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int tmax = TMAX;
    int nx = NX;
    int ny = NY;

    int i __attribute__((annotate("scalar(range(-200, 200) final)")));
    int j __attribute__((annotate("scalar(range(-240, 240) final)")));
    int t __attribute__((annotate("scalar(range(0, 100) final)")));
    //printf("A\n");
    for (i = 0; i < tmax; i++)
        _fict_[i] = (DATA_TYPE) i;
    //printf("B\n");
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
        {
            ex[i][j] = ((DATA_TYPE) i*(j+1)) / nx;
            ey[i][j] = ((DATA_TYPE) i*(j+2)) / ny;
            hz[i][j] = ((DATA_TYPE) i*(j+3)) / nx;
        }
    //printf("C\n");

    for(t = 0; t < _PB_TMAX; t++)
    {
        i=0;
        for (j = 0; j < _PB_NY; j++)
            ey[i][j] = _fict_[t];

    //printf("D\n");
        for (i = 1; i < _PB_NX; i++)
            for (j = 0; j < _PB_NY; j++)
                ey[i][j] = ey[i][j] - SCALAR_VAL(0.5)*(hz[i][j]-hz[i-1][j]);
    //printf("E\n");
        for (i = 0; i < _PB_NX; i++)
            for (j = 1; j < _PB_NY; j++)
                ex[i][j] = ex[i][j] - SCALAR_VAL(0.5)*(hz[i][j]-hz[i][j-1]);
    //printf("F\n");
        for (i = 0; i < _PB_NX - 1; i++)
            for (j = 0; j < _PB_NY - 1; j++)
                hz[i][j] = hz[i][j] - SCALAR_VAL(0.7)*  (ex[i][j+1] - ex[i][j] +
                                                         ey[i+1][j] - ey[i][j]);
    }
    //printf("G\n");

    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();

    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
            if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
            fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ex[i][j]);
        }

    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
            if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
            fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ey[i][j]);
        }

    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
            if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
            fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, hz[i][j]);
        }


    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
