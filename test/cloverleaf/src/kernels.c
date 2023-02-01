// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#include "kernels/kernels.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "data.h"
#include "definitions.h"
#include "utils/timer.h"

void initialise_chunk(int tile) {
  tile_type *tile_ptr = &chunk.tiles[tile];

  double dx = (grid.xmax - grid.xmin) / (double)grid.x_cells;
  double dy = (grid.ymax - grid.ymin) / (double)grid.y_cells;

  double xmin = grid.xmin + dx * (double)(tile_ptr->t_left - 1);
  double ymin = grid.ymin + dy * (double)(tile_ptr->t_bottom - 1);

  kernel_initialise_chunk(
      tile_ptr->t_xmin,
      tile_ptr->t_xmax,
      tile_ptr->t_ymin,
      tile_ptr->t_ymax,
      xmin,
      ymin,
      dx,
      dy,
      tile_ptr->field.vertexx,
      tile_ptr->field.vertexdx,
      tile_ptr->field.vertexy,
      tile_ptr->field.vertexdy,
      tile_ptr->field.cellx,
      tile_ptr->field.celldx,
      tile_ptr->field.celly,
      tile_ptr->field.celldy,
      tile_ptr->field.volume,
      tile_ptr->field.xarea,
      tile_ptr->field.yarea
  );
}

void generate_chunk(int tile) {
  tile_type *tile_ptr = &chunk.tiles[tile];

  double state_density[number_of_states];
  double state_energy[number_of_states];
  double state_xvel[number_of_states];
  double state_yvel[number_of_states];
  double state_xmin[number_of_states];
  double state_xmax[number_of_states];
  double state_ymin[number_of_states];
  double state_ymax[number_of_states];
  double state_radius[number_of_states];
  int state_geometry[number_of_states];

  for (int state = 0; state < number_of_states; state++) {
    state_density[state] = states[state].density;
    state_energy[state] = states[state].energy;
    state_xvel[state] = states[state].xvel;
    state_yvel[state] = states[state].yvel;
    state_xmin[state] = states[state].xmin;
    state_xmax[state] = states[state].xmax;
    state_ymin[state] = states[state].ymin;
    state_ymax[state] = states[state].ymax;
    state_radius[state] = states[state].radius;
    state_geometry[state] = states[state].geometry;
  }

  kernel_generate_chunk(
      tile_ptr->t_xmin,
      tile_ptr->t_xmax,
      tile_ptr->t_ymin,
      tile_ptr->t_ymax,
      tile_ptr->field.vertexx,
      tile_ptr->field.vertexy,
      tile_ptr->field.cellx,
      tile_ptr->field.celly,
      tile_ptr->field.density0,
      tile_ptr->field.energy0,
      tile_ptr->field.xvel0,
      tile_ptr->field.yvel0,
      number_of_states,
      state_density,
      state_energy,
      state_xvel,
      state_yvel,
      state_xmin,
      state_xmax,
      state_ymin,
      state_ymax,
      state_radius,
      state_geometry
  );
}

void ideal_gas(int tile, bool predict) {
  tile_type *tile_ptr = &chunk.tiles[tile];

  kernel_ideal_gas(
      tile_ptr->t_xmin,
      tile_ptr->t_xmax,
      tile_ptr->t_ymin,
      tile_ptr->t_ymax,
      predict ? tile_ptr->field.density1 : tile_ptr->field.density0,
      predict ? tile_ptr->field.energy1 : tile_ptr->field.energy0,
      tile_ptr->field.pressure,
      tile_ptr->field.soundspeed
  );
}

void update_tile_halo(int fields[static NUM_FIELDS], int depth) {
  int t_left, t_right, t_up, t_down;

  for (int tile = 0; tile < tiles_per_chunk; tile++) {
    tile_type *tile_ptr = &chunk.tiles[tile];

    t_up = tile_ptr->tile_neighbours[TILE_TOP];
    t_down = tile_ptr->tile_neighbours[TILE_BOTTOM];

    // Update Top Bottom - Real to Real

    if (t_up != EXTERNAL_TILE) {
      tile_type *tile_top_ptr = &chunk.tiles[t_up];

      kernel_update_tile_halo_t(
          tile_ptr->t_xmin,
          tile_ptr->t_xmax,
          tile_ptr->t_ymin,
          tile_ptr->t_ymax,
          tile_ptr->field.density0,
          tile_ptr->field.energy0,
          tile_ptr->field.pressure,
          tile_ptr->field.viscosity,
          tile_ptr->field.soundspeed,
          tile_ptr->field.density1,
          tile_ptr->field.energy1,
          tile_ptr->field.xvel0,
          tile_ptr->field.yvel0,
          tile_ptr->field.xvel1,
          tile_ptr->field.yvel1,
          tile_ptr->field.vol_flux_x,
          tile_ptr->field.vol_flux_y,
          tile_ptr->field.mass_flux_x,
          tile_ptr->field.mass_flux_y,
          tile_top_ptr->t_xmin,
          tile_top_ptr->t_xmax,
          tile_top_ptr->t_ymin,
          tile_top_ptr->t_ymax,
          tile_top_ptr->field.density0,
          tile_top_ptr->field.energy0,
          tile_top_ptr->field.pressure,
          tile_top_ptr->field.viscosity,
          tile_top_ptr->field.soundspeed,
          tile_top_ptr->field.density1,
          tile_top_ptr->field.energy1,
          tile_top_ptr->field.xvel0,
          tile_top_ptr->field.yvel0,
          tile_top_ptr->field.xvel1,
          tile_top_ptr->field.yvel1,
          tile_top_ptr->field.vol_flux_x,
          tile_top_ptr->field.vol_flux_y,
          tile_top_ptr->field.mass_flux_x,
          tile_top_ptr->field.mass_flux_y,
          fields,
          depth
      );
    }

    if (t_down != EXTERNAL_TILE) {
      tile_type *tile_bottom_ptr = &chunk.tiles[t_down];

      kernel_update_tile_halo_b(
          tile_ptr->t_xmin,
          tile_ptr->t_xmax,
          tile_ptr->t_ymin,
          tile_ptr->t_ymax,
          tile_ptr->field.density0,
          tile_ptr->field.energy0,
          tile_ptr->field.pressure,
          tile_ptr->field.viscosity,
          tile_ptr->field.soundspeed,
          tile_ptr->field.density1,
          tile_ptr->field.energy1,
          tile_ptr->field.xvel0,
          tile_ptr->field.yvel0,
          tile_ptr->field.xvel1,
          tile_ptr->field.yvel1,
          tile_ptr->field.vol_flux_x,
          tile_ptr->field.vol_flux_y,
          tile_ptr->field.mass_flux_x,
          tile_ptr->field.mass_flux_y,
          tile_bottom_ptr->t_xmin,
          tile_bottom_ptr->t_xmax,
          tile_bottom_ptr->t_ymin,
          tile_bottom_ptr->t_ymax,
          tile_bottom_ptr->field.density0,
          tile_bottom_ptr->field.energy0,
          tile_bottom_ptr->field.pressure,
          tile_bottom_ptr->field.viscosity,
          tile_bottom_ptr->field.soundspeed,
          tile_bottom_ptr->field.density1,
          tile_bottom_ptr->field.energy1,
          tile_bottom_ptr->field.xvel0,
          tile_bottom_ptr->field.yvel0,
          tile_bottom_ptr->field.xvel1,
          tile_bottom_ptr->field.yvel1,
          tile_bottom_ptr->field.vol_flux_x,
          tile_bottom_ptr->field.vol_flux_y,
          tile_bottom_ptr->field.mass_flux_x,
          tile_bottom_ptr->field.mass_flux_y,
          fields,
          depth
      );
    }
  }

  // Update Left Right - Ghost, Real, Ghost - > Real

  for (int tile = 0; tile < tiles_per_chunk; tile++) {
    tile_type *tile_ptr = &chunk.tiles[tile];

    int t_left = tile_ptr->tile_neighbours[TILE_LEFT];
    int t_right = tile_ptr->tile_neighbours[TILE_RIGHT];

    if (t_left != EXTERNAL_TILE) {
      tile_type *tile_left_ptr = &chunk.tiles[t_left];

      kernel_update_tile_halo_l(
          tile_ptr->t_xmin,
          tile_ptr->t_xmax,
          tile_ptr->t_ymin,
          tile_ptr->t_ymax,
          tile_ptr->field.density0,
          tile_ptr->field.energy0,
          tile_ptr->field.pressure,
          tile_ptr->field.viscosity,
          tile_ptr->field.soundspeed,
          tile_ptr->field.density1,
          tile_ptr->field.energy1,
          tile_ptr->field.xvel0,
          tile_ptr->field.yvel0,
          tile_ptr->field.xvel1,
          tile_ptr->field.yvel1,
          tile_ptr->field.vol_flux_x,
          tile_ptr->field.vol_flux_y,
          tile_ptr->field.mass_flux_x,
          tile_ptr->field.mass_flux_y,
          tile_left_ptr->t_xmin,
          tile_left_ptr->t_xmax,
          tile_left_ptr->t_ymin,
          tile_left_ptr->t_ymax,
          tile_left_ptr->field.density0,
          tile_left_ptr->field.energy0,
          tile_left_ptr->field.pressure,
          tile_left_ptr->field.viscosity,
          tile_left_ptr->field.soundspeed,
          tile_left_ptr->field.density1,
          tile_left_ptr->field.energy1,
          tile_left_ptr->field.xvel0,
          tile_left_ptr->field.yvel0,
          tile_left_ptr->field.xvel1,
          tile_left_ptr->field.yvel1,
          tile_left_ptr->field.vol_flux_x,
          tile_left_ptr->field.vol_flux_y,
          tile_left_ptr->field.mass_flux_x,
          tile_left_ptr->field.mass_flux_y,
          fields,
          depth
      );
    }

    if (t_right != EXTERNAL_TILE) {
      tile_type *tile_right_ptr = &chunk.tiles[t_right];

      kernel_update_tile_halo_r(
          tile_ptr->t_xmin,
          tile_ptr->t_xmax,
          tile_ptr->t_ymin,
          tile_ptr->t_ymax,
          tile_ptr->field.density0,
          tile_ptr->field.energy0,
          tile_ptr->field.pressure,
          tile_ptr->field.viscosity,
          tile_ptr->field.soundspeed,
          tile_ptr->field.density1,
          tile_ptr->field.energy1,
          tile_ptr->field.xvel0,
          tile_ptr->field.yvel0,
          tile_ptr->field.xvel1,
          tile_ptr->field.yvel1,
          tile_ptr->field.vol_flux_x,
          tile_ptr->field.vol_flux_y,
          tile_ptr->field.mass_flux_x,
          tile_ptr->field.mass_flux_y,
          tile_right_ptr->t_xmin,
          tile_right_ptr->t_xmax,
          tile_right_ptr->t_ymin,
          tile_right_ptr->t_ymax,
          tile_right_ptr->field.density0,
          tile_right_ptr->field.energy0,
          tile_right_ptr->field.pressure,
          tile_right_ptr->field.viscosity,
          tile_right_ptr->field.soundspeed,
          tile_right_ptr->field.density1,
          tile_right_ptr->field.energy1,
          tile_right_ptr->field.xvel0,
          tile_right_ptr->field.yvel0,
          tile_right_ptr->field.xvel1,
          tile_right_ptr->field.yvel1,
          tile_right_ptr->field.vol_flux_x,
          tile_right_ptr->field.vol_flux_y,
          tile_right_ptr->field.mass_flux_x,
          tile_right_ptr->field.mass_flux_y,
          fields,
          depth
      );
    }
  }
}

void update_halo(int fields[static NUM_FIELDS], int depth) {
  double kernel_time;

  if (profiler_on)
    kernel_time = timer();

  update_tile_halo(fields, depth);

  if (profiler_on) {
    profiler.tile_halo_exchange += timer() - kernel_time;
    kernel_time = timer();
  }

  if (chunk.chunk_neighbours[CHUNK_LEFT] == EXTERNAL_FACE || chunk.chunk_neighbours[CHUNK_RIGHT] == EXTERNAL_FACE ||
      chunk.chunk_neighbours[CHUNK_BOTTOM] == EXTERNAL_FACE || chunk.chunk_neighbours[CHUNK_TOP] == EXTERNAL_FACE) {
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
      tile_type *cur_tile = &chunk.tiles[tile];

      kernel_update_halo(
          cur_tile->t_xmin,
          cur_tile->t_xmax,
          cur_tile->t_ymin,
          cur_tile->t_ymax,
          chunk.chunk_neighbours,
          cur_tile->tile_neighbours,
          cur_tile->field.density0,
          cur_tile->field.energy0,
          cur_tile->field.pressure,
          cur_tile->field.viscosity,
          cur_tile->field.soundspeed,
          cur_tile->field.density1,
          cur_tile->field.energy1,
          cur_tile->field.xvel0,
          cur_tile->field.yvel0,
          cur_tile->field.xvel1,
          cur_tile->field.yvel1,
          cur_tile->field.vol_flux_x,
          cur_tile->field.vol_flux_y,
          cur_tile->field.mass_flux_x,
          cur_tile->field.mass_flux_y,
          fields,
          depth
      );
    }
  }

  if (profiler_on)
    profiler.self_halo_exchange += timer() - kernel_time;
}

void field_summary() {
  double vol, mass, ie, ke, press;
  double t_vol, t_mass, t_ie, t_ke, t_press;
  double qa_diff;

  double kernel_time;

  printf("\nTime %4.16f\n", time_val);
  // Print table header row with column width 16
  printf(
      "            %16s%16s%16s%16s%16s%16s%16s\n",
      "Volume",
      "Mass",
      "Density",
      "Pressure",
      "Internal Energy",
      "Kinetic Energy",
      "Total Energy"
  );

  if (profiler_on)
    kernel_time = timer();

  for (int tile = 0; tile < tiles_per_chunk; tile++)
    ideal_gas(tile, false);

  if (profiler_on) {
    profiler.ideal_gas += timer() - kernel_time;
    kernel_time = timer();
  }

  t_vol = 0.0;
  t_mass = 0.0;
  t_ie = 0.0;
  t_ke = 0.0;
  t_press = 0.0;

  for (int tile = 0; tile < tiles_per_chunk; tile++) {
    tile_type *cur_tile = &chunk.tiles[tile];

    kernel_field_summary(
        cur_tile->t_xmin,
        cur_tile->t_xmax,
        cur_tile->t_ymin,
        cur_tile->t_ymax,
        cur_tile->field.volume,
        cur_tile->field.density0,
        cur_tile->field.energy0,
        cur_tile->field.pressure,
        cur_tile->field.xvel0,
        cur_tile->field.yvel0,
        &vol,
        &mass,
        &ie,
        &ke,
        &press
    );

    t_vol += vol;
    t_mass += mass;
    t_ie += ie;
    t_ke += ke;
    t_press += press;
  }

  if (profiler_on)
    profiler.summary += timer() - kernel_time;

  printf(
      "step:%7d%16.4e%16.4e%16.4e%16.4e%16.4e%16.4e%16.4e\n\n",
      step,
      t_vol,
      t_mass,
      t_mass / t_vol,
      t_press / t_vol,
      t_ie,
      t_ke,
      t_ie + t_ke
  );

  if (complete && test_problem >= 1) {
    double ke_constant;
    switch (test_problem) {
      case 1:
        ke_constant = 1.82280367310258;
        break;
      case 2:
        ke_constant = 1.19316898756307;
        break;
      case 3:
        ke_constant = 2.58984003503994;
        break;
      case 4:
        ke_constant = 0.307475452287895;
        break;
      case 5:
        ke_constant = 4.85350315783719;
        break;
      case 6:  // same as test_problem 2 but with 20x20 cells
        ke_constant = 4.78264618380410;
        break;
      case 7:  // same as test_problem 2 but with 19x19 cells
        ke_constant = 5.23176218087986;
        break;
      default:
        ke_constant = 1.0;
        break;
    }

    qa_diff = fabs((100.0 * (t_ke / ke_constant)) - 100.0);

    printf("\nTest problem %4d is within %16.7e%% of the expected solution\n", test_problem, qa_diff);

    if (qa_diff < 0.001) {
      puts("This test is considered PASSED\n");
    } else {
      puts("This test is considered NOT PASSED\n");
    }
  }
}

void visit() {
  // In the original source, this kernel would have printed data to a .vtk file
  // However, since max_task was always equal to 1, the code of this kernel was never actually executed
  // For this reason, the implementation of it has been skipped
}

void viscosity() {
  for (int tile = 0; tile < tiles_per_chunk; tile++) {
    tile_type *cur_tile = &chunk.tiles[tile];

    kernel_viscosity(
        cur_tile->t_xmin,
        cur_tile->t_xmax,
        cur_tile->t_ymin,
        cur_tile->t_ymax,
        cur_tile->field.celldx,
        cur_tile->field.celldy,
        cur_tile->field.density0,
        cur_tile->field.pressure,
        cur_tile->field.viscosity,
        cur_tile->field.xvel0,
        cur_tile->field.yvel0
    );
  }
}

void calc_dt(
    int tile, double *local_dt, char local_control[static 8], double *xl_pos, double *yl_pos, int *jldt, int *kldt
) {
  tile_type *tile_ptr = &chunk.tiles[tile];
  int l_control;
  int small = 0;
  *local_dt = G_BIG;

  kernel_calc_dt(
      tile_ptr->t_xmin,
      tile_ptr->t_xmax,
      tile_ptr->t_ymin,
      tile_ptr->t_ymax,
      dtmin,
      dtc_safe,
      dtu_safe,
      dtv_safe,
      dtdiv_safe,
      tile_ptr->field.xarea,
      tile_ptr->field.yarea,
      tile_ptr->field.cellx,
      tile_ptr->field.celly,
      tile_ptr->field.celldx,
      tile_ptr->field.celldy,
      tile_ptr->field.volume,
      tile_ptr->field.density0,
      tile_ptr->field.energy0,
      tile_ptr->field.pressure,
      tile_ptr->field.viscosity,
      tile_ptr->field.soundspeed,
      tile_ptr->field.xvel0,
      tile_ptr->field.yvel0,
      tile_ptr->field.work_array1,
      local_dt,
      &l_control,
      xl_pos,
      yl_pos,
      jldt,
      kldt,
      small
  );

  switch (l_control) {
    case 1:
      strcpy(local_control, "sound");
      break;
    case 2:
      strcpy(local_control, "xvel");
      break;
    case 3:
      strcpy(local_control, "yvel");
      break;
    case 4:
      strcpy(local_control, "div");
      break;
  }
}

void revert() {
  for (int tile = 0; tile < tiles_per_chunk; tile++) {
    tile_type *tile_ptr = &chunk.tiles[tile];

    kernel_revert(
        tile_ptr->t_xmin,
        tile_ptr->t_xmax,
        tile_ptr->t_ymin,
        tile_ptr->t_ymax,
        tile_ptr->field.density0,
        tile_ptr->field.density1,
        tile_ptr->field.energy0,
        tile_ptr->field.energy1
    );
  }
}

void PdV(bool predict) {
  double kernel_time;
  int fields[NUM_FIELDS];

  if (profiler_on)
    kernel_time = timer();

  for (int tile = 0; tile < tiles_per_chunk; tile++) {
    tile_type *tile_ptr = &chunk.tiles[tile];

    kernel_pdv(
        predict,
        tile_ptr->t_xmin,
        tile_ptr->t_xmax,
        tile_ptr->t_ymin,
        tile_ptr->t_ymax,
        dt,
        tile_ptr->field.xarea,
        tile_ptr->field.yarea,
        tile_ptr->field.volume,
        tile_ptr->field.density0,
        tile_ptr->field.density1,
        tile_ptr->field.energy0,
        tile_ptr->field.energy1,
        tile_ptr->field.pressure,
        tile_ptr->field.viscosity,
        tile_ptr->field.xvel0,
        tile_ptr->field.xvel1,
        tile_ptr->field.yvel0,
        tile_ptr->field.yvel1,
        tile_ptr->field.work_array1
    );
  }

  if (profiler_on)
    profiler.PdV += timer() - kernel_time;

  if (predict) {
    if (profiler_on)
      kernel_time = timer();

    for (int tile = 0; tile < tiles_per_chunk; tile++) {
      ideal_gas(tile, true);
    }

    if (profiler_on)
      profiler.ideal_gas += timer() - kernel_time;

    memset(fields, 0, sizeof(fields));
    fields[FIELD_PRESSURE] = 1;
    update_halo(fields, 1);

    if (profiler_on)
      kernel_time = timer();

    revert();

    if (profiler_on)
      profiler.revert += timer() - kernel_time;
  }
}

void accelerate() {
  double kernel_time;

  if (profiler_on)
    kernel_time = timer();

  for (int tile = 0; tile < tiles_per_chunk; tile++) {
    tile_type *tile_ptr = &chunk.tiles[tile];

    kernel_accelerate(
        tile_ptr->t_xmin,
        tile_ptr->t_xmax,
        tile_ptr->t_ymin,
        tile_ptr->t_ymax,
        dt,
        tile_ptr->field.xarea,
        tile_ptr->field.yarea,
        tile_ptr->field.volume,
        tile_ptr->field.density0,
        tile_ptr->field.pressure,
        tile_ptr->field.viscosity,
        tile_ptr->field.xvel0,
        tile_ptr->field.yvel0,
        tile_ptr->field.xvel1,
        tile_ptr->field.yvel1
    );
  }

  if (profiler_on)
    profiler.acceleration += timer() - kernel_time;
}

void flux_calc() {
  double kernel_time;

  if (profiler_on)
    kernel_time = timer();

  for (int tile = 0; tile < tiles_per_chunk; tile++) {
    tile_type *tile_ptr = &chunk.tiles[tile];

    kernel_flux_calc(
        tile_ptr->t_xmin,
        tile_ptr->t_xmax,
        tile_ptr->t_ymin,
        tile_ptr->t_ymax,
        dt,
        tile_ptr->field.xarea,
        tile_ptr->field.yarea,
        tile_ptr->field.xvel0,
        tile_ptr->field.yvel0,
        tile_ptr->field.xvel1,
        tile_ptr->field.yvel1,
        tile_ptr->field.vol_flux_x,
        tile_ptr->field.vol_flux_y
    );
  }

  if (profiler_on)
    profiler.flux += timer() - kernel_time;
}

void advec_cell(int tile, int sweep_number, int direction) {
  tile_type *tile_ptr = &chunk.tiles[tile];

  kernel_advec_cell(
      tile_ptr->t_xmin,
      tile_ptr->t_xmax,
      tile_ptr->t_ymin,
      tile_ptr->t_ymax,
      direction,
      sweep_number,
      tile_ptr->field.vertexdx,
      tile_ptr->field.vertexdy,
      tile_ptr->field.volume,
      tile_ptr->field.density1,
      tile_ptr->field.energy1,
      tile_ptr->field.mass_flux_x,
      tile_ptr->field.vol_flux_x,
      tile_ptr->field.mass_flux_y,
      tile_ptr->field.vol_flux_y,
      tile_ptr->field.work_array1,
      tile_ptr->field.work_array2,
      tile_ptr->field.work_array3,
      tile_ptr->field.work_array4,
      tile_ptr->field.work_array5,
      tile_ptr->field.work_array6,
      tile_ptr->field.work_array7
  );
}

void advec_mom(int tile, int which_vel, int direction, int sweep_number) {
  tile_type *tile_ptr = &chunk.tiles[tile];

  kernel_advec_mom(
      tile_ptr->t_xmin,
      tile_ptr->t_xmax,
      tile_ptr->t_ymin,
      tile_ptr->t_ymax,
      which_vel == G_XDIR ? tile_ptr->field.xvel1 : tile_ptr->field.yvel1,
      tile_ptr->field.mass_flux_x,
      tile_ptr->field.vol_flux_x,
      tile_ptr->field.mass_flux_y,
      tile_ptr->field.vol_flux_y,
      tile_ptr->field.volume,
      tile_ptr->field.density1,
      tile_ptr->field.work_array1,
      tile_ptr->field.work_array2,
      tile_ptr->field.work_array3,
      tile_ptr->field.work_array4,
      tile_ptr->field.work_array5,
      tile_ptr->field.work_array6,
      tile_ptr->field.celldx,
      tile_ptr->field.celldy,
      which_vel,
      sweep_number,
      direction
  );
}

void advection() {
  int sweep_number, direction, tile;
  int xvel, yvel;
  int fields[NUM_FIELDS];
  double kernel_time;

  sweep_number = 1;
  direction = advect_x ? G_XDIR : G_YDIR;
  xvel = G_XDIR;
  yvel = G_YDIR;

  memset(fields, 0, sizeof(fields));
  fields[FIELD_ENERGY1] = 1;
  fields[FIELD_DENSITY1] = 1;
  fields[FIELD_VOL_FLUX_X] = 1;
  fields[FIELD_VOL_FLUX_Y] = 1;
  update_halo(fields, 2);

  if (profiler_on)
    kernel_time = timer();

  for (tile = 0; tile < tiles_per_chunk; tile++)
    advec_cell(tile, sweep_number, direction);

  if (profiler_on)
    profiler.cell_advection += timer() - kernel_time;

  memset(fields, 0, sizeof(fields));
  fields[FIELD_DENSITY1] = 1;
  fields[FIELD_ENERGY1] = 1;
  fields[FIELD_XVEL1] = 1;
  fields[FIELD_YVEL1] = 1;
  fields[FIELD_MASS_FLUX_X] = 1;
  fields[FIELD_MASS_FLUX_Y] = 1;
  update_halo(fields, 2);

  if (profiler_on)
    kernel_time = timer();

  for (tile = 0; tile < tiles_per_chunk; tile++) {
    advec_mom(tile, xvel, direction, sweep_number);
    advec_mom(tile, yvel, direction, sweep_number);
  }

  if (profiler_on)
    profiler.mom_advection += timer() - kernel_time;

  sweep_number = 2;
  direction = advect_x ? G_YDIR : G_XDIR;

  if (profiler_on)
    kernel_time = timer();

  for (tile = 0; tile < tiles_per_chunk; tile++)
    advec_cell(tile, sweep_number, direction);

  if (profiler_on)
    profiler.cell_advection += timer() - kernel_time;

  memset(fields, 0, sizeof(fields));
  fields[FIELD_DENSITY1] = 1;
  fields[FIELD_ENERGY1] = 1;
  fields[FIELD_XVEL1] = 1;
  fields[FIELD_YVEL1] = 1;
  fields[FIELD_MASS_FLUX_X] = 1;
  fields[FIELD_MASS_FLUX_Y] = 1;
  update_halo(fields, 2);

  if (profiler_on)
    kernel_time = timer();

  for (tile = 0; tile < tiles_per_chunk; tile++) {
    advec_mom(tile, xvel, direction, sweep_number);
    advec_mom(tile, yvel, direction, sweep_number);
  }

  if (profiler_on)
    profiler.mom_advection += timer() - kernel_time;
}

void reset_field() {
  double kernel_time;

  if (profiler_on)
    kernel_time = timer();

  for (int tile = 0; tile < tiles_per_chunk; tile++) {
    tile_type *tile_ptr = &chunk.tiles[tile];

    kernel_reset_field(
        tile_ptr->t_xmin,
        tile_ptr->t_xmax,
        tile_ptr->t_ymin,
        tile_ptr->t_ymax,
        tile_ptr->field.density0,
        tile_ptr->field.density1,
        tile_ptr->field.energy0,
        tile_ptr->field.energy1,
        tile_ptr->field.xvel0,
        tile_ptr->field.xvel1,
        tile_ptr->field.yvel0,
        tile_ptr->field.yvel1
    );
  }

  if (profiler_on)
    profiler.reset += timer() - kernel_time;
}
