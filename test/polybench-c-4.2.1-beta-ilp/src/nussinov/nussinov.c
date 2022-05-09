#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
#   define N 60
#  endif

#  ifdef SMALL_DATASET
#   define N 180
#  endif

#  ifdef MEDIUM_DATASET
#   define N 500
#  endif

#  ifdef LARGE_DATASET
#   define N 2500
#  endif

#  ifdef EXTRALARGE_DATASET
#   define N 5500
#  endif

#define _PB_N N

#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

#define POLYBENCH_DUMP_TARGET stdout


int BENCH_MAIN(){

/* Variable declaration/allocation. */
  PB_STATIC int seq[N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-500, 500) error(1e-100))"))) table[N][N];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int n = N;



    int i __attribute__((annotate("scalar(range(-500, 500) final)")));
    int j __attribute__((annotate("scalar(range(-500, 500) final)")));
    int k;

    //base is AGCT/0..3
    for (i=0; i <n; i++) {
        seq[i] = (int)((i+1)%4);
    }

    for (i=0; i <n; i++)
        for (j=0; j <n; j++)
            table[i][j] = 0;

    for (i = _PB_N-1; i >= 0; i--) {
        for (j=i+1; j<_PB_N; j++) {
            DATA_TYPE __attribute__((annotate("scalar(range(-500, 500))"))) table_i_j = table[i][j];

            if (j-1>=0)
                table_i_j = max_score(table_i_j, table[i][j-1]);
            if (i+1<_PB_N)
                table_i_j = max_score(table_i_j, table[i+1][j]);

            if (j-1>=0 && i+1<_PB_N) {
                /* don't allow adjacent elements to bond */
                if (i<j-1)
                    table_i_j = max_score(table_i_j, table[i+1][j-1]+match(seq[i], seq[j]));
                else
                    table_i_j = max_score(table_i_j, table[i+1][j-1]);
            }

            for (k=i+1; k<j; k++) {
                table_i_j = max_score(table_i_j, table[i][k] + table[k+1][j]);
            }
            table[i][j] = table_i_j;
        }
    }


    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();
    int t=0;
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            if (t % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, table[i][j]);
            t++;
        }
    }





    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
