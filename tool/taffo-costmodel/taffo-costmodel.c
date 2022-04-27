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


#define USE_TIMING
#include "config.h"
#include "timing.h"
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


enum e_ops {
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_DIV,
  OP_SHIFT,
  OP_REM,
  N_OP
};
const char *ops_names[N_OP] = {
  "ADD",
  "SUB",
  "MUL",
  "DIV",
  "SHIFT",
  "REM",
};

enum e_types {
  T_FIX,
  T_FLOAT,
  T_DOUBLE,
  N_T
};
const char *types_names[N_T] = {
  "FIX",
  "FLOAT",
  "DOUBLE"
};

double op_times[N_OP][N_T];
double cast_times[N_OP][N_OP];

/*
enum collecting {
  ADD_FIX = 0,
  ADD_FLOAT,
  ADD_DOUBLE,
  SUB_FIX,
  SUB_FLOAT,
  SUB_DOUBLE,
  MUL_FIX,
  MUL_FLOAT,
  MUL_DOUBLE,
  DIV_FIX,
  DIV_FLOAT,
  DIV_DOUBLE,
  REM_FIX,
  REM_FLOAT,
  REM_DOUBLE,
  LOG_FIX,
  LOG_FLOAT,
  LOG_DOUBLE,
  CAST_FIX_FLOAT,
  CAST_FIX_DOUBLE,
  CAST_FLOAT_FIX,
  CAST_FLOAT_DOUBLE,
  CAST_DOUBLE_FIX,
  CAST_DOUBLE_FLOAT,
  CAST_FIX_FIX,
  COLLECTION_SIZE
};

const char * coll[] = {
  "ADD_FIX",
  "ADD_FLOAT",
  "ADD_DOUBLE",
  "SUB_FIX",
  "SUB_FLOAT",
  "SUB_DOUBLE",
  "MUL_FIX",
  "MUL_FLOAT",
  "MUL_DOUBLE",
  "DIV_FIX",
  "DIV_FLOAT",
  "DIV_DOUBLE",
  "REM_FIX",
  "REM_FLOAT",
  "REM_DOUBLE",
  "CAST_FIX_FLOAT",
  "CAST_FIX_DOUBLE",
  "CAST_FLOAT_FIX",
  "CAST_FLOAT_DOUBLE",
  "CAST_DOUBLE_FIX",
  "CAST_DOUBLE_FLOAT",
  "CAST_FIX_FIX",
  "ERRRRRRRRRRORRRRRRRRRRRR"
};

double times[COLLECTION_SIZE];
*/

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
#define ITIME(OP_ID, OP) do { \
    t1=0; t2=0; t3=0; t4=0; \
    TIME(OP, int32, t3); \
    op_times[OP_ID][T_FIX] = t3; \
    fprintf(stderr, "'%-20s', %16.10f, %16.10f, %16.10f, %16.10f\n", \
            #OP, t1, t2, t3, t4); \
    fflush(stderr); \
  } while (0);

/* multi flt type macro */
#define FTIME(OP_ID, OP) do { \
    t1=0; t2=0; t3=0; t4=0; \
    TIME(OP, flt32, t1); \
    TIME(OP, flt64, t2); \
    op_times[OP_ID][T_FLOAT] = t1; \
    op_times[OP_ID][T_DOUBLE] = t2; \
    fprintf(stderr, "'%-20s', %16.10f, %16.10f, %16.10f, %16.10f\n", \
            #OP, t1, t2, t3, t4); \
    fflush(stderr); \
  } while (0);

/* multi flt function type macro */
#define F_FUN_TIME(OP_ID, DEST, FUN_F, FUN_D, PARM) do { \
    t1=0; t2=0; t3=0; t4=0; \
    TIME(DEST = FUN_F PARM, flt32, t1); \
    TIME(DEST = FUN_D PARM, flt64, t2); \
    op_times[OP_ID][T_FLOAT] = t1; \
    op_times[OP_ID][T_DOUBLE] = t2; \
    fprintf(stderr, "'%-20s', %16.10f, %16.10f, %16.10f, %16.10f\n", \
           #DEST " = " #FUN_F #PARM, t1, t2, t3, t4); \
    fflush(stderr); \
  } while (0);


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

#define max(X, Y) (((X) > (Y)) ? (X) : (Y))
#define CTIME(STYPE_ID, STYPE) do { \
    t1=0; t2=0; t3=0; t4=0; t5=0; \
    CONVTIME(STYPE, flt32, t1); \
    CONVTIME(STYPE, flt64, t2); \
    CONVTIME(STYPE, int32, t3); \
    cast_times[STYPE_ID][T_FLOAT] = t1; \
    cast_times[STYPE_ID][T_DOUBLE] = t2; \
    cast_times[STYPE_ID][T_FIX] = t3; \
    fprintf(stderr, "'%-20s', %16.10f, %16.10f, %16.10f, %16.10f, %16.10f\n", \
           "Cast from " #STYPE, t1, t2, t3, t4, t5); \
    fflush(stderr);\
  } while (0);


int main(void) {
    const size_t memsize = MEMSIZE;
    const int nbrun = NBRUN;
    // "volatile" avoids optimizing out
    volatile unsigned char *_a, *_b, *_c;
    
    timing_cpuclock_init();

    _a = malloc(memsize);
    _b = malloc(memsize);
    _c = malloc(memsize);

    /* initialization */
    srand(0);
    for (size_t i = 0; i < memsize; i++) {
        _a[i] = rand() % (UCHAR_MAX - 1) + 1; // no zero div
        _b[i] = rand() % (UCHAR_MAX - 1) + 1; // no zero div
    }
    printf("%zu Kbytes, median of %d trials\n", memsize / 1000, nbrun);
    float t1, t2, t3, t4, t5;

    printf("Integer Arithmetics\n");
    printf("'%-20s', %16s, %16s, %16s, %16s\n", "Operation", "int8", "int16", "int32", "int64");
    ITIME(OP_ADD, c[i] = a[i] + b[i]);
    ITIME(OP_SUB, c[i] = a[i] - b[i]);
    ITIME(OP_SHIFT, c[i] = a[i] << 3);
    ITIME(OP_MUL, c[i] = a[i] * b[i]);
    ITIME(OP_DIV, c[i] = a[i] / b[i]);
    ITIME(OP_REM, c[i] = a[i] % b[i]);

    printf("Floating-point Arithmetics\n");
    printf("'%-20s', %16s, %16s, %16s, %16s\n", "Operation", "flt32", "flt64", "flt80", "flt128");
    FTIME(OP_ADD, c[i] = a[i] + b[i]);
    FTIME(OP_SUB, c[i] = a[i] + b[i]);
    FTIME(OP_MUL, c[i] = a[i] * b[i]);
    FTIME(OP_DIV, c[i] = a[i] / b[i]);
    F_FUN_TIME(OP_REM, c[i], fmodf, fmod, (a[i], b[i]));

    printf("'%-20s', %16s, %16s, %16s, %16s, %16s\n", " --- To --->", "flt32", "flt64", "int32", "flt80", "flt128");
    CTIME(T_FIX, int32);
    cast_times[T_FIX][T_FLOAT] += op_times[OP_DIV][T_FLOAT];
    cast_times[T_FIX][T_DOUBLE] += op_times[OP_DIV][T_DOUBLE];
    CTIME(T_FLOAT, flt32);
    cast_times[T_FLOAT][T_FIX] += op_times[OP_MUL][T_FLOAT];
    CTIME(T_DOUBLE, flt64);
    cast_times[T_DOUBLE][T_FIX] += op_times[OP_MUL][T_DOUBLE];

    free((void *) _a);
    free((void *) _b);
    free((void *) _c);

    double min = 9E22;
    for (int i = 0; i < N_T; i++) {
      for (int j = 0; j < N_OP; j++) {
        if (op_times[j][i] < min && op_times[j][i] != 0) {
          min = op_times[j][i];
        }
      }
      for (int j = 0; j < N_T; j++) {
        if (cast_times[j][i] < min && cast_times[j][i] != 0) {
          min = cast_times[j][i];
        }
      }
    }

    for (int i = 0; i < N_T; i++) {
      for (int j = 0; j < N_OP; j++) {
        op_times[j][i] /= min;
      }
      for (int j = 0; j < N_T; j++) {
        cast_times[j][i] /= min;
      }
    }

    for (int i = 0; i < N_OP; i++) {
      for (int j = 0; j < N_T; j++) {
        if (op_times[i][j] > 0) {
          printf("%s_%s,\t%lf\n", ops_names[i], types_names[j], op_times[i][j]);
        }
      }
    }
    for (int i=0; i<N_T; i++) {
      for (int j=0; j<N_T; j++) {
        if (cast_times[i][j] > 0) {
          printf("CAST_%s_%s,\t%lf\n", types_names[i], types_names[j], cast_times[i][j]);
        }
      }
    }

    return 0;
}
