#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include "benchmark.h"


enum {
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_DIV,
  OP_REM,
  N_OP
};
const char *ops_names[N_OP] = {
  "ADD",
  "SUB",
  "MUL",
  "DIV",
  "REM",
};

enum {
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

uint64_t op_times[N_OP][N_T];
uint64_t cast_times[N_OP][N_OP];


#define N_ITER (1024*1024*64)

/* Helper macros for infix operators */
#define ADD(x, y)     ((x) + (y))
#define SUB(x, y)     ((x) - (y))
#define MUL(x, y)     ((x) * (y))
#define FIX_MUL(x, y) ((int32_t)(((uint64_t)(x) * (uint64_t)(y)) >> 5))
#define DIV(x, y)     ((x) / (y))
#define FIX_DIV(x, y) ((int32_t)((((uint64_t)(x) << 28) / (uint64_t)(y)) >> 8) + 1)
#define MOD(x, y)     ((x) % (y) + 1)

#define MEASURE_BIN_OP(type_id, type, op_id, op) do { \
    printf("type_id=%-8s op_id=%-8s", #type_id, #op_id); \
    type a=rand(), b=rand(), c=rand(), d=rand(); \
    t_timer timer; \
    timer_start(&timer); \
    for (int i=(N_ITER); i!=0; i--) { \
      d = op(a, b); c = op(d, a); b = op(c, d); a = op(b, c); \
      d = op(a, b); c = op(d, a); b = op(c, d); a = op(b, c); \
      d = op(a, b); c = op(d, a); b = op(c, d); a = op(b, c); \
      d = op(a, b); c = op(d, a); b = op(c, d); a = op(b, c); \
    } \
    use(&a); use(&b); use(&c); use(&d); \
    uint64_t ns = timer_nsSinceStart(&timer); \
    op_times[op_id][type_id] = ns; \
    printf(" %12"PRIu64"\n", ns); \
  } while (0);

#define CAST_FIX_FIX(x)       ((x) >> 20)
#define CAST_FIX_FLOAT(x)     ((float)(x) / (float)0x1000)
#define CAST_FIX_DOUBLE(x)    ((double)(x) / (double)0x1000)
#define CAST_FLOAT_FIX(x)     ((uint32_t)((x) * 0x1000))
#define CAST_FLOAT_DOUBLE(x)  ((double)(x))
#define CAST_DOUBLE_FIX(x)    ((uint32_t)((x) * 0x1000))
#define CAST_DOUBLE_FLOAT(x)  ((float)(x))

#define MEASURE_CAST_OP(type_a_id, type_a, type_b_id, type_b, op, r_op) do { \
    printf("a_type_id=%-8s b_type_id=%-8s ", #type_a_id, #type_b_id); \
    type_a a=rand(); \
    type_b b=rand(); \
    t_timer timer; \
    timer_start(&timer); \
    for (int i=(N_ITER); i!=0; i--) { \
      b = op(a); USE_REGISTER(b); a = r_op(b); USE_REGISTER(a); \
      b = op(a); USE_REGISTER(b); a = r_op(b); USE_REGISTER(a); \
      b = op(a); USE_REGISTER(b); a = r_op(b); USE_REGISTER(a); \
      b = op(a); USE_REGISTER(b); a = r_op(b); USE_REGISTER(a); \
      b = op(a); USE_REGISTER(b); a = r_op(b); USE_REGISTER(a); \
      b = op(a); USE_REGISTER(b); a = r_op(b); USE_REGISTER(a); \
      b = op(a); USE_REGISTER(b); a = r_op(b); USE_REGISTER(a); \
      b = op(a); USE_REGISTER(b); a = r_op(b); USE_REGISTER(a); \
    } \
    use(&a); use(&b); \
    uint64_t ns = timer_nsSinceStart(&timer); \
    cast_times[type_a_id][type_b_id] = ns; \
    cast_times[type_b_id][type_a_id] = ns; \
    printf(" %12"PRIu64"\n", ns); \
  } while (0);

void performBenchmarks(void)
{
  MEASURE_BIN_OP(T_FIX,    uint32_t, OP_ADD, ADD);
  MEASURE_BIN_OP(T_FLOAT,  float,    OP_ADD, ADD);
  MEASURE_BIN_OP(T_DOUBLE, double,   OP_ADD, ADD);
  MEASURE_BIN_OP(T_FIX,    uint32_t, OP_SUB, ADD);
  MEASURE_BIN_OP(T_FLOAT,  float,    OP_SUB, ADD);
  MEASURE_BIN_OP(T_DOUBLE, double,   OP_SUB, ADD);
  MEASURE_BIN_OP(T_FIX,    uint32_t, OP_MUL, FIX_MUL);
  MEASURE_BIN_OP(T_FLOAT,  float,    OP_MUL, MUL);
  MEASURE_BIN_OP(T_DOUBLE, double,   OP_MUL, MUL);
  MEASURE_BIN_OP(T_FIX,    uint32_t, OP_DIV, FIX_DIV);
  MEASURE_BIN_OP(T_FLOAT,  float,    OP_DIV, DIV);
  MEASURE_BIN_OP(T_DOUBLE, double,   OP_DIV, DIV);
  MEASURE_BIN_OP(T_FIX,    uint32_t, OP_REM, MOD);
  MEASURE_BIN_OP(T_FLOAT,  float,    OP_REM, fmodf);
  MEASURE_BIN_OP(T_DOUBLE, double,   OP_REM, fmod);

  MEASURE_CAST_OP(T_FIX,    uint32_t, T_FIX,    uint32_t, CAST_FIX_FIX,      CAST_FIX_FIX);
  MEASURE_CAST_OP(T_FIX,    uint32_t, T_FLOAT,  float,    CAST_FIX_FLOAT,    CAST_FLOAT_FIX);
  MEASURE_CAST_OP(T_FIX,    uint32_t, T_DOUBLE, double,   CAST_FIX_DOUBLE,   CAST_DOUBLE_FIX);
  MEASURE_CAST_OP(T_FLOAT,  uint32_t, T_DOUBLE, double,   CAST_FLOAT_DOUBLE, CAST_DOUBLE_FLOAT);
}


void printModel(FILE *fout)
{
  uint64_t min = op_times[OP_ADD][T_FIX];

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

  for (int i = 0; i < N_OP; i++) {
    for (int j = 0; j < N_T; j++) {
      if (op_times[i][j] > 0) {
        double ratio = (double)op_times[i][j] / (double)min;
        fprintf(fout, "%s_%s,\t%lf\n", ops_names[i], types_names[j], ratio);
      }
    }
  }
  for (int i=0; i<N_T; i++) {
    for (int j=0; j<N_T; j++) {
      if (cast_times[i][j] > 0) {
        double ratio = (double)cast_times[i][j] / (double)min;
        fprintf(fout, "CAST_%s_%s,\t%lf\n", types_names[i], types_names[j], ratio);
      }
    }
  }
}


int main(int argc, char *argv[])
{
  performBenchmarks();
  printModel(stdout);
  return 0;
}
