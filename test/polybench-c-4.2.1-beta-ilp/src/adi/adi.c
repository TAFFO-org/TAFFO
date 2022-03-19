#include <stdio.h>
#include <stdlib.h>
#include "instrument.h"
#define DATA_TYPE double
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
#   define TSTEPS 20
#   define N 20
#  endif

#  ifdef SMALL_DATASET
#   define TSTEPS 40
#   define N 60
#  endif

#  ifdef MEDIUM_DATASET
#   define TSTEPS 100
#   define N 200
#  endif

#  ifdef LARGE_DATASET
#   define TSTEPS 500
#   define N 1000
#  endif

#  ifdef EXTRALARGE_DATASET
#   define TSTEPS 1000
#   define N 2000
#  endif

#   define _PB_TSTEPS TSTEPS
#   define _PB_N N

int BENCH_MAIN(int argc, char** argv)
{

  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) DX, DY, DT;
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) B1, B2;
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) mul1, mul2;
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) a, b, c, d, e, f;


  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-60,60) final error(1e-100))")))u[N][N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-2,2) final error(1e-100))")))v[N][N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-1,1) final error(1e-100))")))p[N][N];
  PB_STATIC DATA_TYPE __attribute__((annotate("scalar(range(-500,500) final error(1e-100))")))q[N][N];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int n = N;
    int tsteps = TSTEPS;






    int i __attribute__((annotate("scalar(range(-4000, 4000) final)")));
    int j __attribute__((annotate("scalar(range(-4000, 4000) final)")));

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            u[i][j] =  (DATA_TYPE)(i + n-j) / n;
        }



    int t;



    DX = (1.0)/_PB_N;
    DY = (1.0)/_PB_N;
    DT = (1.0)/_PB_TSTEPS;
    B1 = (2.0);
    B2 = (1.0);
    mul1 = B1 * DT / (DX * DX);
    mul2 = B2 * DT / (DY * DY);

    a = -mul1 /  SCALAR_VAL(2.0);
    b = SCALAR_VAL(1.0)+mul1;
    c = a;
    d = -mul2 / SCALAR_VAL(2.0);
    e = SCALAR_VAL(1.0)+mul2;
    f = d;


    for (t=1; t<=_PB_TSTEPS; t++) {
        //Column Sweep
        for (i=1; i<_PB_N-1; i++) {
            j=0; //TRYING TO FIX
            v[j][i] = SCALAR_VAL(1.0);
            p[i][j] = SCALAR_VAL(0.0);
            q[i][j] = v[j][i];

            for (j=1; j<_PB_N-1; j++) {

                p[i][j] = -c / (a*p[i][j-1]+b); //FIXME: here is the error

                q[i][j] = (-d*u[j][i-1]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[j][i] - f*u[j][i+1]-a*q[i][j-1]);

                q[i][j] /= (a*p[i][j-1]+b);

            }

            j=_PB_N-1;
            v[j][i] = SCALAR_VAL(1.0);

            for (j=_PB_N-2; j>=1; j--) {
                v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
            }

        }

        //Row Sweep
        for (i=1; i<_PB_N-1; i++) {
            j=0; //TRYING TO FIX
            u[i][j] = SCALAR_VAL(1.0);
            p[i][j] = SCALAR_VAL(0.0);
            q[i][j] = u[i][j];
            for (j=1; j<_PB_N-1; j++) {
                p[i][j] = -f / (d*p[i][j-1]+e); //FIXME: here is the error
                q[i][j] = (-a*v[i-1][j]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][j] - c*v[i+1][j]-d*q[i][j-1]);
                q[i][j] /= (d*p[i][j-1]+e);
            }
            j=_PB_N-1;
            u[i][j] = SCALAR_VAL(1.0);
            for (j=_PB_N-2; j>=1; j--) {
                u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
            }
        }

    }


    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            if ((i * n + j) % 20 == 0) fprintf(stdout, "\n");
            fprintf (stdout, DATA_PRINTF_MODIFIER, u[i][j]);
        }



    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
