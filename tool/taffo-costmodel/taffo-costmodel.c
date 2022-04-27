/*
 * Copyright (c) 2014-2015, Nicolas Limare <nicolas@limare.net>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

/**
 * @file time_arit.c
 * @brief timing arithmetic and math operations
 *
 * @author Nicolas Limare <nicolas@limare.net>
 */

#include "xutils.h"

#define USE_TIMING

#include "config.h"
#include "timing.h"
#include "collector.h"

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;

typedef float flt32;
typedef double flt64;

#ifdef NOFLT80
#define RUNFLT80 0
typedef float flt80;
#else
#define RUNFLT80 1
typedef __float80 flt80;
#endif

#ifdef NOFLT128
#define RUNFLT128 0
typedef float flt128;
#else
#define RUNFLT128 1
typedef __float128 flt128;
#endif

/**
 * floating-point comparison, for qsort()
 */
static int cmpf(const void *a, const void *b) {
    float fa = *(float *) a;
    float fb = *(float *) b;
    return (fa > fb) - (fa < fb);
}

/* single type macro */
#define TIME(OP, TYPE, T) {                        \
    float cpuclock[nbrun];                        \
    volatile TYPE * a = (TYPE *) _a;                \
    volatile TYPE * b = (TYPE *) _b;                \
    volatile TYPE * c = (TYPE *) _c;                \
    const size_t nbops = memsize / sizeof(TYPE);            \
    for (int n = 0; n < nbrun; n++) {                \
        TIMING_CPUCLOCK_START(0);                    \
        for (size_t i = 0; i < nbops; i++)                \
        OP;                            \
        TIMING_CPUCLOCK_TOGGLE(0);                    \
        cpuclock[n] = TIMING_CPUCLOCK_S(0);                \
    }                                \
    qsort(cpuclock, nbrun, sizeof(float), &cmpf);            \
    T = nbops / 1E6 / cpuclock[nbrun/2];                \
    T = 1000.0/T; \
    (TYPE *)a;\
    (TYPE *)b;\
    (TYPE *)c;\
    }

/* multi int type macro */
#define ITIME(OP) {                        \
    t1=0; t2=0; t3=0; t4=0;                    \
    TIME(OP, int8, t1);                    \
    TIME(OP, int16, t2);                    \
    TIME(OP, int32, t3);                    \
    TIME(OP, int64, t4);                    \
    fprintf(stderr, "'%-20s', %16.10f, %16.10f, %16.10f, %16.10f\n",    \
           #OP, t1, t2, t3, t4);                \
    fflush(stderr);\
    }


/* multi flt type macro */
#define FTIME(OP) {                        \
    t1=0; t2=0; t3=0; t4=0;                \
    TIME(OP, flt32, t1);                    \
    TIME(OP, flt64, t2);                    \
    if (RUNFLT80) TIME(OP, flt80, t3);            \
    if (RUNFLT128) TIME(OP, flt128, t4);            \
    fprintf(stderr, "'%-20s', %16.10f, %16.10f, %16.10f, %16.10f\n",    \
           #OP, t1, t2, t3, t4);            \
    fflush(stderr);\
    }

#define max(X, Y) (((X) > (Y)) ? (X) : (Y))
#define CTIME(STYPE) {                        \
    t1=0; t2=0; t3=0; t4=0; t5=0;                \
    CONVTIME(STYPE, flt32, t1);                    \
    CONVTIME(STYPE, flt64, t2);                    \
    CONVTIME(STYPE, int32, t3);            \
    if (RUNFLT80) CONVTIME(STYPE, flt80, t4);            \
    if (RUNFLT128) CONVTIME(STYPE, flt128, t5);            \
    fprintf(stderr, "'%-20s', %16.10f, %16.10f, %16.10f, %16.10f, %16.10f\n",    \
           "Cast from " #STYPE, t1, t2, t3, t4, t5);            \
    fflush(stderr);\
    }

/* single type macro */
#define CONVTIME(STYPE, DTYPE, T) {                        \
    float cpuclock[nbrun];                        \
    volatile STYPE * a = (STYPE *) _a;                \
    volatile DTYPE * b = (DTYPE *) _b;                \
    const size_t nbops = memsize / max(sizeof(STYPE), sizeof(DTYPE));            \
    for (int n = 0; n < nbrun; n++) {                \
        TIMING_CPUCLOCK_START(0);                    \
        for (size_t i = 0; i < nbops; i++)                \
        b[i] = (DTYPE) a[i];                            \
        TIMING_CPUCLOCK_TOGGLE(0);                    \
        cpuclock[n] = TIMING_CPUCLOCK_S(0);                \
    }                                \
    qsort(cpuclock, nbrun, sizeof(float), &cmpf);            \
    T = nbops / 1E6 / cpuclock[nbrun/2];                \
    T = 1000.0/T; \
    }


int main(void) {
    const size_t memsize = MEMSIZE;
    const int nbrun = NBRUN;
    // "volatile" avoids optimizing out
    volatile unsigned char *_a, *_b, *_c;

    _a = xmalloc(memsize);
    _b = xmalloc(memsize);
    _c = xmalloc(memsize);

    /* initialization */
    srand(0);
    for (size_t i = 0; i < memsize; i++) {
        _a[i] = rand() % (UCHAR_MAX - 1) + 1; // no zero div
        _b[i] = rand() % (UCHAR_MAX - 1) + 1; // no zero div
    }
    INFO("%zu Kbytes, median of %d trials",
         memsize / 1000, nbrun);
    float t1, t2, t3, t4, t5;

    INFO("Integer Arithmetics");
    fprintf(stderr, "'%-20s', %16s, %16s, %16s, %16s\n", "Operation", "int8", "int16", "int32", "int64");
    ITIME(c[i] = a[i] + b[i]);
    times[ADD_FIX] = t3;
    times[SUB_FIX] = t3;
    ITIME(c[i] = a[i] & b[i]);
    ITIME(c[i] = a[i] | b[i]);
    ITIME(c[i] = a[i] ^ b[i]);
    ITIME(c[i] = a[i] << 3);
    times[CAST_FIX_FIX] = t3;
    ITIME(c[i] = a[i] * b[i]);
    times[MUL_FIX] = t4;
    ITIME(c[i] = a[i] / b[i]);
    times[DIV_FIX] = t4;
    ITIME(c[i] = a[i] % b[i]);
    times[REM_FIX] = t3;

    INFO("Floating-point Arithmetics");
    fprintf(stderr, "'%-20s', %16s, %16s, %16s, %16s\n", "Operation", "flt32", "flt64", "flt80", "flt128");
    FTIME(c[i] = a[i] + b[i]);
    times[ADD_FLOAT] = t1;
    times[SUB_FLOAT] = t1;
    times[ADD_DOUBLE] = t2;
    times[SUB_DOUBLE] = t2;
    FTIME(c[i] = a[i] * b[i]);
    times[MUL_FLOAT] = t1;
    times[MUL_DOUBLE] = t2;
    FTIME(c[i] = a[i] / b[i]);
    times[DIV_FLOAT] = t1;
    times[DIV_DOUBLE] = t2;
    FTIME(c[i] = fmod(a[i], b[i]));
    times[REM_FLOAT] = t1;
    times[REM_DOUBLE] = t2;

    fprintf(stderr, "'%-20s', %16s, %16s, %16s, %16s, %16s\n", " --- To --->", "flt32", "flt64", "int32", "flt80", "flt128");
    CTIME(int32);
    times[CAST_FIX_FLOAT] = t1+times[DIV_FLOAT];
    times[CAST_FIX_DOUBLE] = t2+times[DIV_DOUBLE];
    CTIME(flt32);
    times[CAST_FLOAT_FIX] = t3+times[MUL_FLOAT];
    times[CAST_FLOAT_DOUBLE] = t2;
    CTIME(flt64);
    times[CAST_DOUBLE_FLOAT] = t1;
    times[CAST_DOUBLE_FIX] = t3+times[MUL_DOUBLE];

    xfree((void *) _a);
    xfree((void *) _b);
    xfree((void *) _c);

    double min = 9E22;
    for (int i = 0; i < COLLECTION_SIZE; i++) {
        if (times[i] < min) {
            min = times[i];
        }
    }

    for (int i = 0; i < COLLECTION_SIZE; i++) {
        times[i] /= min;
    }

    for (int i = 0; i < COLLECTION_SIZE; i++) {
        printf("%s,\t%lf\n", coll[i], times[i]);
    }

    return EXIT_SUCCESS;
}
