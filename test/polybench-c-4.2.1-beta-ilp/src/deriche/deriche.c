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
#   define W 64
#   define H 64
#  endif

#  ifdef SMALL_DATASET
#   define W 192
#   define H 128
#  endif

#  ifdef MEDIUM_DATASET
#   define W 720
#   define H 480
#  endif

#  ifdef LARGE_DATASET
#   define W 4096
#   define H 2160
#  endif

#  ifdef EXTRALARGE_DATASET
#   define W 7680
#   define H 4320
#  endif

#   define _PB_W W
#   define _PB_H H

#  define DATA_PRINTF_MODIFIER "%0.16lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)

#define DATA_TYPE double
#define ALPHA 0.25

int BENCH_MAIN(){


  PB_STATIC DATA_TYPE   __attribute__((annotate("scalar(error(1e-100))"))) imgIn[W][H];
  PB_STATIC DATA_TYPE  __attribute__((annotate("scalar(error(1e-100))"))) imgOut[W][H];
  PB_STATIC DATA_TYPE  __attribute__((annotate("scalar(error(1e-100))"))) _y1[W][H];
  PB_STATIC DATA_TYPE  __attribute__((annotate("scalar(error(1e-100))"))) y2[W][H];

    TAFFO_DUMPCONFIG();
    TIMING_CPUCLOCK_START();
    /* Retrieve problem size. */
    int w = W;
    int h = H;

    /* Variable declaration/allocation. */
    DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) alpha;

    int __attribute__((annotate("scalar(range(-192, 192) final )"))) i;
    int __attribute__((annotate("scalar(range(-128, 128) final )"))) j;


    alpha=0.25; //parameter of the filter

    //input should be between 0 and 1 (grayscale image pixel)
    for (i = 0; i < w; i++)
        for (j = 0; j < h; j++)
            imgIn[i][j] = (DATA_TYPE) ((313*i+991*j)%65536) / 65535.0f;


    DATA_TYPE __attribute__((annotate("scalar(range(-1, 1) final error(1e-100))"))) xm1, tm1, ym1;
    DATA_TYPE __attribute__((annotate("scalar(range(-1, 1) final error(1e-100))"))) ym2;
    DATA_TYPE __attribute__((annotate("scalar(range(-1, 1) final error(1e-100))"))) xp1, xp2;
    DATA_TYPE __attribute__((annotate("scalar(range(-1, 1) final error(1e-100))"))) tp1, tp2;
    DATA_TYPE __attribute__((annotate("scalar(range(-1, 1) final error(1e-100))"))) yp1, yp2;

    DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) k;
    DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) a1, a2, a3, a4, a5, a6, a7, a8;
    DATA_TYPE __attribute__((annotate("scalar(error(1e-100))"))) b1, b2, c1, c2;

    k = (SCALAR_VAL(1.0)-EXP_FUN(-ALPHA))*(SCALAR_VAL(1.0)-EXP_FUN(-ALPHA))/(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*ALPHA*EXP_FUN(-ALPHA)-EXP_FUN(SCALAR_VAL(2.0)*ALPHA));
    a1 = a5 = k;
    a2 = a6 = k*EXP_FUN(-ALPHA)*(ALPHA-SCALAR_VAL(1.0));
    a3 = a7 = k*EXP_FUN(-ALPHA)*(ALPHA+SCALAR_VAL(1.0));
    a4 = a8 = -k*EXP_FUN(SCALAR_VAL(-2.0)*ALPHA);
    b1 =  POW_FUN(SCALAR_VAL(2.0),-ALPHA);
    b2 = -EXP_FUN(SCALAR_VAL(-2.0)*ALPHA);
    c1 = c2 = 1;

    /*printf("%.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f ",
           xm1, tm1, ym1, ym2, xp1, xp2, tp1, tp2, yp1, yp2, k, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, c1, c2);*/

    for (i=0; i<_PB_W; i++) {
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        xm1 = SCALAR_VAL(0.0);
        for (j=0; j<_PB_H; j++) {
            _y1[i][j] = a1*imgIn[i][j] + a2*xm1 + b1*ym1 + b2*ym2;
            xm1 = imgIn[i][j];
            ym2 = ym1;
            ym1 = _y1[i][j];
        }

    }



    for (i=0; i<_PB_W; i++) {
        yp1 = SCALAR_VAL(0.0);
        yp2 = SCALAR_VAL(0.0);
        xp1 = SCALAR_VAL(0.0);
        xp2 = SCALAR_VAL(0.0);
        for (j=_PB_H-1; j>=0; j--) {
            y2[i][j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2;
            xp2 = xp1;
            xp1 = imgIn[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for (i=0; i<_PB_W; i++)
        for (j=0; j<_PB_H; j++) {
            imgOut[i][j] = c1 * (_y1[i][j] + y2[i][j]);
        }

    for (j=0; j<_PB_H; j++) {
        tm1 = SCALAR_VAL(0.0);
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        for (i=0; i<_PB_W; i++) {
            _y1[i][j] = a5*imgOut[i][j] + a6*tm1 + b1*ym1 + b2*ym2;
            tm1 = imgOut[i][j];
            ym2 = ym1;
            ym1 = _y1 [i][j];
        }
    }


    for (j=0; j<_PB_H; j++) {
        tp1 = SCALAR_VAL(0.0);
        tp2 = SCALAR_VAL(0.0);
        yp1 = SCALAR_VAL(0.0);
        yp2 = SCALAR_VAL(0.0);
        for (i=_PB_W-1; i>=0; i--) {
            y2[i][j] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2;
            tp2 = tp1;
            tp1 = imgOut[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for (i=0; i<_PB_W; i++)
        for (j=0; j<_PB_H; j++)
            imgOut[i][j] = c2*(_y1[i][j] + y2[i][j]);


    TIMING_CPUCLOCK_TOGGLE();
    TIMING_CPUCLOCK_PRINT();


    for (i = 0; i < w; i++)
        for (j = 0; j < h; j++) {
            if ((i * h + j) % 20 == 0) fprintf(stdout, "\n");
            fprintf(stdout, "%.16lf ", imgOut[i][j]);
        }



    return 0;
}

#ifdef __TAFFO__
void *__taffo_vra_starting_function = BENCH_MAIN;
#endif
