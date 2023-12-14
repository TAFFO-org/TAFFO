/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* fdtd-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "fdtd-2d.h"


/* Array initialization. */
static
void init_array (int tmax,
		 int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int i __attribute__((annotate("scalar(range(-" PB_XSTR(NX) ", " PB_XSTR(NX) ") final)")));
  int j __attribute__((annotate("scalar(range(-" PB_XSTR(NY) ", " PB_XSTR(NY) ") final)")));

  for (i = 0; i < tmax; i++)
    _fict_[i] = (DATA_TYPE) i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      {
	ex[i][j] = ((DATA_TYPE) i*(j+1)) / nx;
	ey[i][j] = ((DATA_TYPE) i*(j+2)) / ny;
	hz[i][j] = ((DATA_TYPE) i*(j+3)) / nx;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("ex");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ex[i][j]);
    }
  POLYBENCH_DUMP_END("ex");
  POLYBENCH_DUMP_FINISH;

  POLYBENCH_DUMP_BEGIN("ey");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ey[i][j]);
    }
  POLYBENCH_DUMP_END("ey");

  POLYBENCH_DUMP_BEGIN("hz");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, hz[i][j]);
    }
  POLYBENCH_DUMP_END("hz");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_fdtd_2d(int tmax,
		    int nx,
		    int ny,
		    DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int t, i, j;

#pragma scop

  for(t = 0; t < _PB_TMAX; t++)
    {
      for (j = 0; j < _PB_NY; j++)
	ey[0][j] = _fict_[t];
      for (i = 1; i < _PB_NX; i++)
	for (j = 0; j < _PB_NY; j++)
	  ey[i][j] = ey[i][j] - SCALAR_VAL(0.5)*(hz[i][j]-hz[i-1][j]);
      for (i = 0; i < _PB_NX; i++)
	for (j = 1; j < _PB_NY; j++)
	  ex[i][j] = ex[i][j] - SCALAR_VAL(0.5)*(hz[i][j]-hz[i][j-1]);
      for (i = 0; i < _PB_NX - 1; i++)
	for (j = 0; j < _PB_NY - 1; j++)
	  hz[i][j] = hz[i][j] - SCALAR_VAL(0.7)*  (ex[i][j+1] - ex[i][j] +
				       ey[i+1][j] - ey[i][j]);
    }

#pragma endscop
}


int BENCH_MAIN(int argc, char** argv)
{
  /* Retrieve problem size. */
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE __attribute__((annotate("target('ex') scalar(range(" PB_XSTR(VAR_ex_MIN) "," PB_XSTR(VAR_ex_MAX) "))"))),NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE __attribute__((annotate("target('ey') scalar(range(" PB_XSTR(VAR_ey_MIN) "," PB_XSTR(VAR_ey_MAX) "))"))),NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE __attribute__((annotate("target('hz') scalar(range(" PB_XSTR(VAR_hz_MIN) "," PB_XSTR(VAR_hz_MAX) "))"))),NX,NY,nx,ny);
  POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE __attribute__((annotate("scalar(range(" PB_XSTR(VAR_fict_MIN) "," PB_XSTR(VAR_fict_MAX) "))"))),TMAX,tmax);

  for (int benchmark_i = 0; benchmark_i < BENCH_NUM_ITERATIONS; benchmark_i++) {
      /* Initialize array(s). */
      init_array(tmax, nx, ny,
                 POLYBENCH_ARRAY(ex),
                 POLYBENCH_ARRAY(ey),
                 POLYBENCH_ARRAY(hz),
                 POLYBENCH_ARRAY(_fict_));

      srand(POLYBENCH_RANDOM_SEED);
      randomize_2d(NX, NY, ex, POLYBENCH_RANDOMIZE_RANGE);
      randomize_2d(NX, NY, ey, POLYBENCH_RANDOMIZE_RANGE);
      randomize_2d(NX, NY, hz, POLYBENCH_RANDOMIZE_RANGE);
//      randomize_1d(TMAX, _fict_, POLYBENCH_RANDOMIZE_RANGE);

#if SCALING_FACTOR!=1
      scale_2d(NX, NY, POLYBENCH_ARRAY(ex), SCALING_FACTOR);
      scale_2d(NX, NY, POLYBENCH_ARRAY(ey), SCALING_FACTOR);
      scale_2d(NX, NY, POLYBENCH_ARRAY(hz), SCALING_FACTOR);
      scale_1d(TMAX, POLYBENCH_ARRAY(_fict_), SCALING_FACTOR);
#endif

#ifdef COLLECT_STATS
      stats_header();
      stats_2d("ex", NX, NY, POLYBENCH_ARRAY(ex));
      stats_2d("ey", NX, NY, POLYBENCH_ARRAY(ey));
      stats_2d("hz", NX, NY, POLYBENCH_ARRAY(hz));
      stats_1d("fict", TMAX, POLYBENCH_ARRAY(_fict_));
#endif

      /* Start timer. */
      polybench_start_instruments;

      /* Run kernel. */
      kernel_fdtd_2d(tmax, nx, ny,
                     POLYBENCH_ARRAY(ex),
                     POLYBENCH_ARRAY(ey),
                     POLYBENCH_ARRAY(hz),
                     POLYBENCH_ARRAY(_fict_));


      /* Stop and print timer. */
      polybench_stop_instruments;
      polybench_print_instruments;
  }

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(ex),
				    POLYBENCH_ARRAY(ey),
				    POLYBENCH_ARRAY(hz)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(ex);
  POLYBENCH_FREE_ARRAY(ey);
  POLYBENCH_FREE_ARRAY(hz);
  POLYBENCH_FREE_ARRAY(_fict_);

  return 0;
}
