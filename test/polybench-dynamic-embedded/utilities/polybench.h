/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/*
 * polybench.h: this file is part of PolyBench/C
 *
 * Polybench header for instrumentation.
 *
 * Programs must be compiled with `-I utilities utilities/polybench.c'
 *
 * Optionally, one can define:
 *
 * -DPOLYBENCH_TIME, to report the execution time,
 *   OR (exclusive):
 * -DPOLYBENCH_PAPI, to use PAPI H/W counters (defined in polybench.c)
 *
 *
 * See README or utilities/polybench.c for additional options.
 *
 */
#ifndef POLYBENCH_H
# define POLYBENCH_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <sys/resource.h>
#include <sched.h>
#include <math.h>
#include <stdbool.h>
#include <errno.h>

#ifndef BENCH_MAIN
#define BENCH_MAIN main
#endif

#ifndef BENCH_NUM_ITERATIONS
#define BENCH_NUM_ITERATIONS 1
#endif

#ifndef POLYBENCH_RANDOMIZE_ENABLED
#define POLYBENCH_RANDOMIZE_ENABLED 0
#endif

#ifndef POLYBENCH_RANDOM_SEED
#define POLYBENCH_RANDOM_SEED 1
#endif

#ifndef POLYBENCH_RANDOMIZE_RANGE
#define POLYBENCH_RANDOMIZE_RANGE 0
#endif

#ifdef DATA_TYPE_IS_INT
#  define DATA_TYPE int
#  define DATA_PRINTF_MODIFIER "%d "
#endif

#ifdef DATA_TYPE_IS_FLOAT
#  define DATA_TYPE  float
#  define DATA_PRINTF_MODIFIER "%0.16f "
#  define SCALAR_VAL(x) x##f
#  define SQRT_FUN(x) sqrtf_PB(x)
#  define EXP_FUN(x) expf(x)
#  define POW_FUN(x,y) powf(x,y)
# endif

#ifdef DATA_TYPE_IS_DOUBLE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.16lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)
# endif

#define PB_STR(x) #x
#define PB_XSTR(x) PB_STR(x)

/* Array padding. By default, none is used. */
# ifndef POLYBENCH_PADDING_FACTOR
/* default: */
#  define POLYBENCH_PADDING_FACTOR 0
# endif

/* Inter-array padding, for use with . By default, none is used. */
# ifndef POLYBENCH_INTER_ARRAY_PADDING_FACTOR
/* default: */
#  define POLYBENCH_INTER_ARRAY_PADDING_FACTOR 0
#  undef POLYBENCH_ENABLE_INTARRAY_PAD
# else
#  define POLYBENCH_ENABLE_INTARRAY_PAD
# endif


/* C99 arrays in function prototype. By default, do not use. */
# ifdef POLYBENCH_USE_C99_PROTO
#  define POLYBENCH_C99_SELECT(x,y) y
# else
/* default: */
#  define POLYBENCH_C99_SELECT(x,y) x
# endif


/* Scalar loop bounds in SCoPs. By default, use parametric loop bounds. */
# ifdef POLYBENCH_USE_SCALAR_LB
#  define POLYBENCH_LOOP_BOUND(x,y) x
# else
/* default: */
#  define POLYBENCH_LOOP_BOUND(x,y) y
# endif

/* Use the 'restrict' keyword to declare that the different arrays do not
 * alias. By default, we do not use it as it is only supported in C99 and
 * even here several compilers do not properly get it.
 */
# ifdef POLYBENCH_USE_RESTRICT
#  define POLYBENCH_RESTRICT restrict
# else
/* default: */
#  define POLYBENCH_RESTRICT
# endif

/* Macros to reference an array. Generic for heap and stack arrays
   (C99).  Each array dimensionality has his own macro, to be used at
   declaration or as a function argument.
   Example:
   int b[x] => POLYBENCH_1D_ARRAY(b, x)
   int A[N][N] => POLYBENCH_2D_ARRAY(A, N, N)
*/
# ifndef POLYBENCH_STACK_ARRAYS
#  define POLYBENCH_ARRAY(x) *x
#  ifdef POLYBENCH_ENABLE_INTARRAY_PAD
#   define POLYBENCH_FREE_ARRAY(x) polybench_free_data((void*)x);
#  else
#   define POLYBENCH_FREE_ARRAY(x) free((void*)x);
#  endif
#  define POLYBENCH_DECL_VAR(x) (*x)
# else
#  define POLYBENCH_ARRAY(x) x
#  define POLYBENCH_FREE_ARRAY(x)
#  define POLYBENCH_DECL_VAR(x) x
# endif
/* Macros for using arrays in the function prototypes. */
# define POLYBENCH_1D(var, dim1,ddim1) var[POLYBENCH_RESTRICT POLYBENCH_C99_SELECT(dim1,ddim1) + POLYBENCH_PADDING_FACTOR]
# define POLYBENCH_2D(var, dim1, dim2, ddim1, ddim2) var[POLYBENCH_RESTRICT POLYBENCH_C99_SELECT(dim1,ddim1) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim2,ddim2) + POLYBENCH_PADDING_FACTOR]
# define POLYBENCH_3D(var, dim1, dim2, dim3, ddim1, ddim2, ddim3) var[POLYBENCH_RESTRICT POLYBENCH_C99_SELECT(dim1,ddim1) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim2,ddim2) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim3,ddim3) + POLYBENCH_PADDING_FACTOR]
# define POLYBENCH_4D(var, dim1, dim2, dim3, dim4, ddim1, ddim2, ddim3, ddim4) var[POLYBENCH_RESTRICT POLYBENCH_C99_SELECT(dim1,ddim1) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim2,ddim2) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim3,ddim3) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim4,ddim4) + POLYBENCH_PADDING_FACTOR]
# define POLYBENCH_5D(var, dim1, dim2, dim3, dim4, dim5, ddim1, ddim2, ddim3, ddim4, ddim5) var[POLYBENCH_RESTRICT POLYBENCH_C99_SELECT(dim1,ddim1) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim2,ddim2) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim3,ddim3) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim4,ddim4) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim5,ddim5) + POLYBENCH_PADDING_FACTOR]
/* Macros for using arrays within the functions. */
# define POLYBENCH_1D_F(var, dim1,ddim1) var[POLYBENCH_C99_SELECT(dim1,ddim1) + POLYBENCH_PADDING_FACTOR]
# define POLYBENCH_2D_F(var, dim1, dim2, ddim1, ddim2) var[POLYBENCH_C99_SELECT(dim1,ddim1) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim2,ddim2) + POLYBENCH_PADDING_FACTOR]
# define POLYBENCH_3D_F(var, dim1, dim2, dim3, ddim1, ddim2, ddim3) var[POLYBENCH_C99_SELECT(dim1,ddim1) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim2,ddim2) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim3,ddim3) + POLYBENCH_PADDING_FACTOR]
# define POLYBENCH_4D_F(var, dim1, dim2, dim3, dim4, ddim1, ddim2, ddim3, ddim4) var[POLYBENCH_C99_SELECT(dim1,ddim1) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim2,ddim2) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim3,ddim3) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim4,ddim4) + POLYBENCH_PADDING_FACTOR]
# define POLYBENCH_5D_F(var, dim1, dim2, dim3, dim4, dim5, ddim1, ddim2, ddim3, ddim4, ddim5) var[POLYBENCH_C99_SELECT(dim1,ddim1) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim2,ddim2) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim3,ddim3) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim4,ddim4) + POLYBENCH_PADDING_FACTOR][POLYBENCH_C99_SELECT(dim5,ddim5) + POLYBENCH_PADDING_FACTOR]


/* Macros to allocate heap arrays.
   Example:
   polybench_alloc_2d_array(N, M, double) => allocates N x M x sizeof(double)
					  and returns a pointer to the 2d array
 */
# define POLYBENCH_ALLOC_1D_ARRAY(n1, type)	\
  (type(*)[n1 + POLYBENCH_PADDING_FACTOR])polybench_alloc_data (n1 + POLYBENCH_PADDING_FACTOR, sizeof(type))
# define POLYBENCH_ALLOC_2D_ARRAY(n1, n2, type)		\
  (type(*)[n1 + POLYBENCH_PADDING_FACTOR][n2 + POLYBENCH_PADDING_FACTOR])polybench_alloc_data ((n1 + POLYBENCH_PADDING_FACTOR) * (n2 + POLYBENCH_PADDING_FACTOR), sizeof(type))
# define POLYBENCH_ALLOC_3D_ARRAY(n1, n2, n3, type)		\
  (type(*)[n1 + POLYBENCH_PADDING_FACTOR][n2 + POLYBENCH_PADDING_FACTOR][n3 + POLYBENCH_PADDING_FACTOR])polybench_alloc_data ((n1 + POLYBENCH_PADDING_FACTOR) * (n2 + POLYBENCH_PADDING_FACTOR) * (n3 + POLYBENCH_PADDING_FACTOR), sizeof(type))
# define POLYBENCH_ALLOC_4D_ARRAY(n1, n2, n3, n4, type)			\
  (type(*)[n1 + POLYBENCH_PADDING_FACTOR][n2 + POLYBENCH_PADDING_FACTOR][n3 + POLYBENCH_PADDING_FACTOR][n4 + POLYBENCH_PADDING_FACTOR])polybench_alloc_data ((n1 + POLYBENCH_PADDING_FACTOR) * (n2 + POLYBENCH_PADDING_FACTOR) * (n3 + POLYBENCH_PADDING_FACTOR) * (n4 + POLYBENCH_PADDING_FACTOR), sizeof(type))
# define POLYBENCH_ALLOC_5D_ARRAY(n1, n2, n3, n4, n5, type)		\
  (type(*)[n1 + POLYBENCH_PADDING_FACTOR][n2 + POLYBENCH_PADDING_FACTOR][n3 + POLYBENCH_PADDING_FACTOR][n4 + POLYBENCH_PADDING_FACTOR][n5 + POLYBENCH_PADDING_FACTOR])polybench_alloc_data ((n1 + POLYBENCH_PADDING_FACTOR) * (n2 + POLYBENCH_PADDING_FACTOR) * (n3 + POLYBENCH_PADDING_FACTOR) * (n4 + POLYBENCH_PADDING_FACTOR) * (n5 + POLYBENCH_PADDING_FACTOR), sizeof(type))

/* Macros for array declaration. */
# ifndef POLYBENCH_STACK_ARRAYS
#  define POLYBENCH_1D_ARRAY_DECL(var, type, dim1, ddim1)		\
  type POLYBENCH_1D_F(POLYBENCH_DECL_VAR(var), dim1, ddim1); \
  var = POLYBENCH_ALLOC_1D_ARRAY(POLYBENCH_C99_SELECT(dim1, ddim1), type);
#  define POLYBENCH_2D_ARRAY_DECL(var, type, dim1, dim2, ddim1, ddim2)	\
  type POLYBENCH_2D_F(POLYBENCH_DECL_VAR(var), dim1, dim2, ddim1, ddim2); \
  var = POLYBENCH_ALLOC_2D_ARRAY(POLYBENCH_C99_SELECT(dim1, ddim1), POLYBENCH_C99_SELECT(dim2, ddim2), type);
#  define POLYBENCH_3D_ARRAY_DECL(var, type, dim1, dim2, dim3, ddim1, ddim2, ddim3) \
  type POLYBENCH_3D_F(POLYBENCH_DECL_VAR(var), dim1, dim2, dim3, ddim1, ddim2, ddim3); \
  var = POLYBENCH_ALLOC_3D_ARRAY(POLYBENCH_C99_SELECT(dim1, ddim1), POLYBENCH_C99_SELECT(dim2, ddim2), POLYBENCH_C99_SELECT(dim3, ddim3), type);
#  define POLYBENCH_4D_ARRAY_DECL(var, type, dim1, dim2, dim3, dim4, ddim1, ddim2, ddim3, ddim4) \
  type POLYBENCH_4D_F(POLYBENCH_DECL_VAR(var), dim1, dim2, dim3, dim4, ddim1, ddim2, ddim3, ddim4); \
  var = POLYBENCH_ALLOC_4D_ARRAY(POLYBENCH_C99_SELECT(dim1, ddim1), POLYBENCH_C99_SELECT(dim2, ddim2), POLYBENCH_C99_SELECT(dim3, ddim3), POLYBENCH_C99_SELECT(dim4, ddim4), type);
#  define POLYBENCH_5D_ARRAY_DECL(var, type, dim1, dim2, dim3, dim4, dim5, ddim1, ddim2, ddim3, ddim4, ddim5) \
  type POLYBENCH_5D_F(POLYBENCH_DECL_VAR(var), dim1, dim2, dim3, dim4, dim5, ddim1, ddim2, ddim3, ddim4, ddim5); \
  var = POLYBENCH_ALLOC_5D_ARRAY(POLYBENCH_C99_SELECT(dim1, ddim1), POLYBENCH_C99_SELECT(dim2, ddim2), POLYBENCH_C99_SELECT(dim3, ddim3), POLYBENCH_C99_SELECT(dim4, ddim4), POLYBENCH_C99_SELECT(dim5, ddim5), type);
# else
#  define POLYBENCH_1D_ARRAY_DECL(var, type, dim1, ddim1)		\
  type POLYBENCH_1D_F(POLYBENCH_DECL_VAR(var), dim1, ddim1);
#  define POLYBENCH_2D_ARRAY_DECL(var, type, dim1, dim2, ddim1, ddim2)	\
  type POLYBENCH_2D_F(POLYBENCH_DECL_VAR(var), dim1, dim2, ddim1, ddim2);
#  define POLYBENCH_3D_ARRAY_DECL(var, type, dim1, dim2, dim3, ddim1, ddim2, ddim3) \
  type POLYBENCH_3D_F(POLYBENCH_DECL_VAR(var), dim1, dim2, dim3, ddim1, ddim2, ddim3);
#  define POLYBENCH_4D_ARRAY_DECL(var, type, dim1, dim2, dim3, dim4, ddim1, ddim2, ddim3, ddim4) \
  type POLYBENCH_4D_F(POLYBENCH_DECL_VAR(var), dim1, dim2, dim3, dim4, ddim1, ddim2, ddim3, ddim4);
#  define POLYBENCH_5D_ARRAY_DECL(var, type, dim1, dim2, dim3, dim4, dim5, ddim1, ddim2, ddim3, ddim4, ddim5) \
  type POLYBENCH_5D_F(POLYBENCH_DECL_VAR(var), dim1, dim2, dim3, dim4, dim5, ddim1, ddim2, ddim3, ddim4, ddim5);
# endif


/* Dead-code elimination macros. Use argc/argv for the run-time check. */
# ifndef POLYBENCH_DUMP_ARRAYS
#  define POLYBENCH_DCE_ONLY_CODE
# else
#  define POLYBENCH_DCE_ONLY_CODE
# endif

#define POLYBENCH_DUMP_TARGET stderr
#define POLYBENCH_DUMP_START    fprintf(POLYBENCH_DUMP_TARGET, "\n==BEGIN_DUMP_ARRAYS==\n")
#define POLYBENCH_DUMP_FINISH   fprintf(POLYBENCH_DUMP_TARGET, "\n==END_DUMP_ARRAYS==\n")
//#define POLYBENCH_DUMP_BEGIN(s) fprintf(POLYBENCH_DUMP_TARGET, "begin dump: %s", s)
//#define POLYBENCH_DUMP_END(s)   fprintf(POLYBENCH_DUMP_TARGET, "\nend   dump: %s\n", s)
//#define POLYBENCH_DUMP_START
//#define POLYBENCH_DUMP_FINISH
#define POLYBENCH_DUMP_BEGIN(s) 
#define POLYBENCH_DUMP_END(s)   

# define polybench_prevent_dce(func)		\
  POLYBENCH_DCE_ONLY_CODE			\
  func


/* Performance-related instrumentation. See polybench.c */
# define polybench_start_instruments
# define polybench_stop_instruments
# define polybench_print_instruments


/* Timing support. */
# if defined(POLYBENCH_TIME) || defined(POLYBENCH_GFLOPS)
#  undef polybench_start_instruments
#  undef polybench_stop_instruments
#  undef polybench_print_instruments
#  define polybench_start_instruments polybench_timer_start();
#  define polybench_stop_instruments polybench_timer_stop();
#  define polybench_print_instruments polybench_timer_print();
extern double polybench_program_total_flops;
extern void polybench_timer_start();
extern void polybench_timer_stop();
extern void polybench_timer_print();
# endif

/* Function prototypes. */
extern void* polybench_alloc_data(unsigned long long int n, int elt_size);
extern void polybench_free_data(void* ptr);

/* PolyBench internal functions that should not be directly called by */
/* the user, unless when designing customized execution profiling */
/* approaches. */
extern void polybench_flush_cache();
extern void polybench_prepare_instruments();

#ifndef SCALING_FACTOR
#define SCALING_FACTOR 1
#endif

/*
 * Allocation table, to enable inter-array padding. All data allocated
 * with polybench_alloc_data should be freed with polybench_free_data.
 *
 */
#define NB_INITIAL_TABLE_ENTRIES 512
struct polybench_data_ptrs
{
  void** user_view;
  void** real_ptr;
  int nb_entries;
  int nb_avail_entries;
};
static struct polybench_data_ptrs* _polybench_alloc_table = NULL;
static size_t polybench_inter_array_padding_sz = 0;

/*
 * These functions are used only if the user defines a specific
 * inter-array padding. It grows a global structure,
 * _polybench_alloc_table, which keeps track of the data allocated via
 * polybench_alloc_data (on which inter-array padding is applied), so
 * that the original, non-shifted pointer can be recovered when
 * calling polybench_free_data.
 *
 */
#ifdef POLYBENCH_ENABLE_INTARRAY_PAD
static
    void grow_alloc_table()
{
  if (_polybench_alloc_table == NULL ||
      (_polybench_alloc_table->nb_entries % NB_INITIAL_TABLE_ENTRIES) != 0 ||
      _polybench_alloc_table->nb_avail_entries != 0)
  {
    /* Should never happen if the API is properly used. */
    fprintf (stderr, "[ERROR] Inter-array padding requires to use polybench_alloc_data and polybench_free_data\n");
    exit (1);
  }
  size_t sz = _polybench_alloc_table->nb_entries;
  sz += NB_INITIAL_TABLE_ENTRIES;
  _polybench_alloc_table->user_view =
      realloc (_polybench_alloc_table->user_view, sz * sizeof(void*));
  assert(_polybench_alloc_table->user_view != NULL);
  _polybench_alloc_table->real_ptr =
      realloc (_polybench_alloc_table->real_ptr, sz * sizeof(void*));
  assert(_polybench_alloc_table->real_ptr != NULL);
  _polybench_alloc_table->nb_avail_entries = NB_INITIAL_TABLE_ENTRIES;
}

static
    void* register_padded_pointer(void* ptr, size_t orig_sz, size_t padded_sz)
{
  if (_polybench_alloc_table == NULL)
  {
    fprintf (stderr, "[ERROR] Inter-array padding requires to use polybench_alloc_data and polybench_free_data\n");
    exit (1);
  }
  if (_polybench_alloc_table->nb_avail_entries == 0)
    grow_alloc_table ();
  int id = _polybench_alloc_table->nb_entries++;
  _polybench_alloc_table->real_ptr[id] = ptr;
  _polybench_alloc_table->user_view[id] = ptr + (padded_sz - orig_sz);

  return _polybench_alloc_table->user_view[id];
}


static
    void
    free_data_from_alloc_table (void* ptr)
{
  if (_polybench_alloc_table != NULL && _polybench_alloc_table->nb_entries > 0)
  {
    int i;
    for (i = 0; i < _polybench_alloc_table->nb_entries; ++i)
      if (_polybench_alloc_table->user_view[i] == ptr ||
          _polybench_alloc_table->real_ptr[i] == ptr)
        break;
    if (i != _polybench_alloc_table->nb_entries)
    {
      free (_polybench_alloc_table->real_ptr[i]);
      for (; i < _polybench_alloc_table->nb_entries - 1; ++i)
      {
        _polybench_alloc_table->user_view[i] =
            _polybench_alloc_table->user_view[i + 1];
        _polybench_alloc_table->real_ptr[i] =
            _polybench_alloc_table->real_ptr[i + 1];
      }
      _polybench_alloc_table->nb_entries--;
      _polybench_alloc_table->nb_avail_entries++;
      if (_polybench_alloc_table->nb_entries == 0)
      {
        free (_polybench_alloc_table->user_view);
        free (_polybench_alloc_table->real_ptr);
        free (_polybench_alloc_table);
        _polybench_alloc_table = NULL;
      }
    }
  }
}

static
    void check_alloc_table_state()
{
  if (_polybench_alloc_table == NULL)
  {
    _polybench_alloc_table = (struct polybench_data_ptrs*)
        malloc (sizeof(struct polybench_data_ptrs));
    assert(_polybench_alloc_table != NULL);
    _polybench_alloc_table->user_view =
        (void**) malloc (sizeof(void*) * NB_INITIAL_TABLE_ENTRIES);
    assert(_polybench_alloc_table->user_view != NULL);
    _polybench_alloc_table->real_ptr =
        (void**) malloc (sizeof(void*) * NB_INITIAL_TABLE_ENTRIES);
    assert(_polybench_alloc_table->real_ptr != NULL);
    _polybench_alloc_table->nb_entries = 0;
    _polybench_alloc_table->nb_avail_entries = NB_INITIAL_TABLE_ENTRIES;
  }
}

#endif // !POLYBENCH_ENABLE_INTARRAY_PAD


static void* xmalloc(size_t alloc_sz)
{
  void* ret = NULL;
  /* By default, post-pad the arrays. Safe behavior, but likely useless. */
  polybench_inter_array_padding_sz += POLYBENCH_INTER_ARRAY_PADDING_FACTOR;
  size_t padded_sz = alloc_sz + polybench_inter_array_padding_sz;
  ret = memalign(4096, padded_sz);
  if (! ret)
  {
    fprintf (stderr, "[PolyBench] memalign: cannot allocate memory\n");
    if (errno == ENOMEM) {
      fprintf (stderr, "[PolyBench] There was insufficient memory available to satisfy the request: %zu\n", padded_sz);
    }
    exit (1);
  }
  /* Safeguard: this is invoked only if polybench.c has been compiled
     with inter-array padding support from polybench.h. If so, move
     the starting address of the allocation and return it to the
     user. The original pointer is registered in an allocation table
     internal to polybench.c. Data must then be freed using
     polybench_free_data, which will inspect the allocation table to
     free the original pointer.*/
#ifdef POLYBENCH_ENABLE_INTARRAY_PAD
  /* This moves the 'ret' pointer by (padded_sz - alloc_sz) positions, and
  registers it in the lookup table for future free using
  polybench_free_data. */
  ret = register_padded_pointer(ret, alloc_sz, padded_sz);
#endif

  return ret;
}


void polybench_free_data(void* ptr)
{
#ifdef POLYBENCH_ENABLE_INTARRAY_PAD
  free_data_from_alloc_table (ptr);
#else
  free (ptr);
#endif
}


void* polybench_alloc_data(unsigned long long int n, int elt_size)
{
#ifdef POLYBENCH_ENABLE_INTARRAY_PAD
  check_alloc_table_state ();
#endif

  /// FIXME: detect overflow!
  size_t val = n;
  val *= elt_size;
  void* ret = xmalloc (val);

  return ret;
}

#ifdef COLLECT_STATS

void stats_header() {
  fprintf (POLYBENCH_DUMP_TARGET, "var_scaling_factor,var_name,var_min,var_max,var_isnan,var_isinf\n");
}

void stats_scalar(char* name, DATA_TYPE val) {
  fprintf (POLYBENCH_DUMP_TARGET, "%d,%s,%f,%f,%d,%d\n",
          SCALING_FACTOR, name, val, val, isnan(val), isinf(val));
}

void stats_1d(char* name, int n, DATA_TYPE val[n]) {
  float min = val[0];
  float max = val[0];
  bool is_nan = false;
  bool is_inf = false;
  for (int i = 0; i < n; i++) {
    float elem = val[i];
    if (!isfinite(elem)) {
      is_nan |= isnan(elem);
      is_inf |= isinf(elem);
      continue;
    }
    if (min > elem) min = elem;
    if (max < elem) max = elem;
  }
  fprintf (POLYBENCH_DUMP_TARGET, "%d,%s,%f,%f,%d,%d\n",
          SCALING_FACTOR, name, min, max, is_nan, is_inf);
}

void stats_2d(char* name, int n, int m, DATA_TYPE val[n][m]) {
  float min = val[0][0];
  float max = val[0][0];
  bool is_nan = false;
  bool is_inf = false;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      float elem = val[i][j];
      if (!isfinite(elem)) {
        is_nan |= isnan(elem);
        is_inf |= isinf(elem);
        continue;
      }
      if (min > elem) min = elem;
      if (max < elem) max = elem;
    }
  }
  fprintf (POLYBENCH_DUMP_TARGET, "%d,%s,%f,%f,%d,%d\n",
          SCALING_FACTOR, name, min, max, is_nan, is_inf);
}

void stats_3d(char* name, int n, int m, int p, DATA_TYPE val[n][m][p]) {
  float min = val[0][0][0];
  float max = val[0][0][0];
  bool is_nan = false;
  bool is_inf = false;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < p; k++) {
        float elem = val[i][j][k];
        if (!isfinite(elem)) {
          is_nan |= isnan(elem);
          is_inf |= isinf(elem);
          continue;
        }
        if (min > elem) min = elem;
        if (max < elem) max = elem;
      }
    }
  }
  fprintf (POLYBENCH_DUMP_TARGET, "%d,%s,%f,%f,%d,%d\n",
          SCALING_FACTOR, name, min, max, is_nan, is_inf);
}

#endif

#if POLYBENCH_RANDOMIZE_ENABLED > 0

extern float RandomNumber(float min, float max);

void randomize_scalar(float *val) {
  *val += *val * RandomNumber(-POLYBENCH_RANDOMIZE_RANGE, POLYBENCH_RANDOMIZE_RANGE);
}

void randomize_1d(int n, float val[n]) {
  for (int i = 0; i < n; i++) {
    val[i] += val[i] * RandomNumber(-POLYBENCH_RANDOMIZE_RANGE, POLYBENCH_RANDOMIZE_RANGE);
  }
}

void randomize_2d(int n, int m, float val[n][m]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      val[i][j] += val[i][j] * RandomNumber(-POLYBENCH_RANDOMIZE_RANGE, POLYBENCH_RANDOMIZE_RANGE);
    }
  }
}

void randomize_3d(int n, int m, int p, float val[n][m][p]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < p; k++) {
        val[i][j][k] += val[i][j][k] * RandomNumber(-POLYBENCH_RANDOMIZE_RANGE, POLYBENCH_RANDOMIZE_RANGE);
      }
    }
  }
}

#else

void inline __attribute__((always_inline)) randomize_scalar(float *val) {}

void inline __attribute__((always_inline)) randomize_1d(int n, float val[n]) {}

void inline __attribute__((always_inline)) randomize_2d(int n, int m, float val[n][m]) {}

void inline __attribute__((always_inline)) randomize_3d(int n, int m, int p, float val[n][m][p]) {}

#endif



#endif /* !POLYBENCH_H */
