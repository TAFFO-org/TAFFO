// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clover.h"
#include "data.h"
#include "definitions.h"
#include "kernels.h"
#include "report.h"
#include "user_callbacks.h"
#include "utils/math.h"
#include "utils/timer.h"

void timestep();

void hydro() {
  int loc = 1;
  double timerstart, wall_clock, step_clock;
  double grind_time, cells, rstep;
  double step_time, step_grind;
  double first_step, second_step;
  double kerner_total;

  hydro_init();

  timerstart = timer();

  while (true) {
    step_time = timer();
    step++;

    hydro_foreach_step(step);

    timestep();
    PdV(true);
    accelerate();
    PdV(false);
    flux_calc();
    advection();
    reset_field();

    advect_x = !advect_x;

    time_val += dt;

    if (summary_frequency != 0 && step % summary_frequency == 0)
      field_summary();

    if (visit_frequency != 0 && step % visit_frequency == 0)
      visit();

    // Sometimes there can be a significant start up cost that appears in the first step.
    // Sometimes it is due to the number of MPI tasks, or OpenCL kernel compilation.
    // On the short test runs, this can skew the results, so should be taken into account
    // in recorded run times.
    if (step == 1)
      first_step = timer() - step_time;
    else if (step == 2)
      second_step = timer() - step_time;

    if (time_val + G_SMALL > end_time || step >= end_step) {
      complete = true;
      field_summary();
      if (visit_frequency != 0)
        visit();

      wall_clock = timer() - timerstart;
      if (parallel.boss) {
        fprintf(g_out, "\nCalculation completed\n");
        fprintf(g_out, "Clover is finishing\n");
        fprintf(g_out, "Wall clock     %.16f\n", wall_clock);
        fprintf(g_out, "First step overhead   %.16f\n", first_step - second_step);
      }
      printf("Wall clock    %.16f\n", wall_clock);
      printf("First step overhead   %.16f\n", first_step - second_step);

      if (profiler_on) {
        kerner_total = profiler.timestep + profiler.ideal_gas + profiler.viscosity + profiler.PdV + profiler.revert +
                       profiler.acceleration + profiler.flux + profiler.cell_advection + profiler.mom_advection +
                       profiler.reset + profiler.summary + profiler.visit + profiler.tile_halo_exchange +
                       profiler.self_halo_exchange + profiler.mpi_halo_exchange;

        if (parallel.boss) {
          const char *fmt = "\n%-22s:%16.4f%16.4f\n";

          fprintf(g_out, "\n%-22s%16s%20s\n", "Profiler Output", "Time", "Percentage");
          fprintf(g_out, fmt, "Timestep", profiler.timestep, profiler.timestep / wall_clock * 100);
          fprintf(g_out, fmt, "Ideal Gas", profiler.ideal_gas, profiler.ideal_gas / wall_clock * 100);
          fprintf(g_out, fmt, "Viscosity", profiler.viscosity, profiler.viscosity / wall_clock * 100);
          fprintf(g_out, fmt, "PdV", profiler.PdV, profiler.PdV / wall_clock * 100);
          fprintf(g_out, fmt, "Revert", profiler.revert, profiler.revert / wall_clock * 100);
          fprintf(g_out, fmt, "Acceleration", profiler.acceleration, profiler.acceleration / wall_clock * 100);
          fprintf(g_out, fmt, "Fluxes", profiler.flux, profiler.flux / wall_clock * 100);
          fprintf(g_out, fmt, "Cell advection", profiler.cell_advection, profiler.cell_advection / wall_clock * 100);
          fprintf(g_out, fmt, "Momentum advection", profiler.mom_advection, profiler.mom_advection / wall_clock * 100);
          fprintf(g_out, fmt, "Reset", profiler.reset, profiler.reset / wall_clock * 100);
          fprintf(g_out, fmt, "Summary", profiler.summary, profiler.summary / wall_clock * 100);
          fprintf(g_out, fmt, "Visit", profiler.visit, profiler.visit / wall_clock * 100);
          fprintf(
              g_out,
              fmt,
              "Tile halo exchange",
              profiler.tile_halo_exchange,
              profiler.tile_halo_exchange / wall_clock * 100
          );
          fprintf(
              g_out,
              fmt,
              "Self halo exchange",
              profiler.self_halo_exchange,
              profiler.self_halo_exchange / wall_clock * 100
          );
          fprintf(
              g_out, fmt, "MPI halo exchange", profiler.mpi_halo_exchange, profiler.mpi_halo_exchange / wall_clock * 100
          );
          fprintf(g_out, fmt, "Total", kerner_total, kerner_total / wall_clock * 100);
          fprintf(g_out, fmt, "The Rest", wall_clock - kerner_total, (wall_clock - kerner_total) / wall_clock * 100);
        }
      }

      hydro_done();
      clover_finalize();
      break;
    }

    if (parallel.boss) {
      wall_clock = timer() - timerstart;
      step_clock = timer() - step_time;

      fprintf(g_out, "Wall clock    %.16f\n", wall_clock);
      printf("Wall clock    %.16f\n", wall_clock);

      cells = grid.x_cells * grid.y_cells;
      rstep = step + 1;
      grind_time = wall_clock / (rstep * cells);
      step_grind = step_clock / cells;

      fprintf(g_out, "Average time per cell    %.16e\n", grind_time);
      fprintf(g_out, "Step time per cell       %.16e\n", step_grind);
      printf("Average time per cell    %.16e\n", grind_time);
      printf("Step time per cell       %.16e\n", step_grind);
    }
  }
}

void timestep() {
  int tile;
  int jldt, kldt;

  double dtlp;
  double x_pos, y_pos, xl_pos, yl_pos;
  double kernel_time;

  char dt_control[8], dtl_control[8];

  int small;
  int fields[NUM_FIELDS];

  dt = G_BIG;
  small = 0;

  if (profiler_on)
    kernel_time = timer();

  for (tile = 0; tile < tiles_per_chunk; tile++) {
    ideal_gas(tile, false);
  }

  if (profiler_on)
    profiler.ideal_gas += timer() - kernel_time;

  memset(fields, 0, NUM_FIELDS * sizeof(int));
  fields[FIELD_PRESSURE] = 1;
  fields[FIELD_ENERGY0] = 1;
  fields[FIELD_DENSITY0] = 1;
  fields[FIELD_XVEL0] = 1;
  fields[FIELD_YVEL0] = 1;
  update_halo(fields, 1);

  if (profiler_on)
    kernel_time = timer();

  viscosity();

  if (profiler_on)
    profiler.viscosity += timer() - kernel_time;

  memset(fields, 0, NUM_FIELDS * sizeof(int));
  fields[FIELD_VISCOSITY] = 1;
  update_halo(fields, 1);

  if (profiler_on)
    kernel_time = timer();

  for (tile = 0; tile < tiles_per_chunk; tile++) {
    calc_dt(tile, &dtlp, dtl_control, &xl_pos, &yl_pos, &jldt, &kldt);

    if (dtlp <= dt) {
      dt = dtlp;
      strcpy(dt_control, dtl_control);
      x_pos = xl_pos;
      y_pos = yl_pos;
      jdt = jldt;
      kdt = kldt;
    }
  }

  dt = min(dt, min((dtold * dtrise), dtmax));

  if (profiler_on)
    profiler.timestep += timer() - kernel_time;

  if (dt < dtmin)
    small = 1;

  if (parallel.boss) {
    const char *format = "Step %7d time %.7lf control %10s  timestep  %.2e%8d, %8d x  %.2e y  %.2e\n";
    fprintf(g_out, format, step, time_val, dt_control, dt, jdt, kdt, x_pos, y_pos);
    printf(format, step, time_val, dt_control, dt, jdt, kdt, x_pos, y_pos);
  }

  if (small == 1)
    report_error("timestep", "small timestep");

  dtold = dt;
}
