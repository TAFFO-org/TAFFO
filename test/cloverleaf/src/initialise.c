// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clover.h"
#include "data.h"
#include "definitions.h"
#include "kernels.h"
#include "parse.h"
#include "report.h"
#include "utils/math.h"
#include "utils/string.h"

void read_input();

void start();

/**
 * @file allocate.c
 */
extern void build_field();

/**
 * @brief Top level initialisation routine
 * @details Checks for the user input and either invokes the input reader or switches to the internal test problem. It
 * processes the input and strips comments before writing a final input file. It then calls the start routine.
 */
void initialise() {
  FILE *uin = NULL, *out_unit = NULL;

  if (parallel.boss) {
    errno = 0;
    g_out = fopen("clover.out", "w");
    if (errno != 0) {
      g_out = stderr;
      printf("Redirecting file output to stderr\n");
    }

    fprintf(g_out, "Clover Version %f\nMPI Version\nTask Count %d\n", G_VERSION, parallel.max_task);

    puts("Output file clover.out opened. All output will go there.");
  }

  read_input();

  step = 0;

  start();

  if (parallel.boss)
    fputs("Starting the calculation\n", g_out);
}

/**
 * @brief Reads the user input
 * @details Reads and parses the user input from the processed file and sets the variables used in the generation phase.
 * Default values are also set here.
 */
void read_input() {
  int state, stat, state_max;
  double dx, dy;
  char *word;

  test_problem = 7;
  state_max = 0;

  grid.xmin = 0.0;   // xmin
  grid.ymin = 0.0;   // ymin
  grid.ymax = 10.0;  // ymax
  grid.xmax = 10.0;  // xmax

  grid.x_cells = 19;  // x_cells
  grid.y_cells = 19;  // y_cells

  end_time = 10.0;
  end_step = 87;
  complete = false;

  visit_frequency = 0;
  summary_frequency = 10;

  tiles_per_chunk = 1;

  dtinit = 0.04;      // initial_timestep
  dtmax = 0.04;       // max_timestep
  dtmin = 0.0000001;  // min_timestep
  dtrise = 1.5;       // timestep_rise
  dtc_safe = 0.7;
  dtu_safe = 0.5;
  dtv_safe = 0.5;
  dtdiv_safe = 0.7;

  use_fortran_kernels = false;
  use_C_kernels = true;
  use_OA_kernels = false;
  profiler_on = false;
  profiler.timestep = 0.0;
  profiler.acceleration = 0.0;
  profiler.PdV = 0.0;
  profiler.cell_advection = 0.0;
  profiler.mom_advection = 0.0;
  profiler.viscosity = 0.0;
  profiler.ideal_gas = 0.0;
  profiler.visit = 0.0;
  profiler.summary = 0.0;
  profiler.reset = 0.0;
  profiler.revert = 0.0;
  profiler.flux = 0.0;
  profiler.tile_halo_exchange = 0.0;
  profiler.self_halo_exchange = 0.0;
  profiler.mpi_halo_exchange = 0.0;

  if (parallel.boss)
    fputs("Reading input file\n\n", g_out);

  number_of_states = 2;
  states = calloc(number_of_states, sizeof(state_type));

  states[0] = (state_type){.defined = true, .density = 0.2, .energy = 1.0};

  states[1] = (state_type){
      .defined = true,
      .density = 1.0,
      .energy = 2.5,
      .geometry = G_RECT,
      .xmin = 0.0,
      .xmax = 5.0,
      .ymin = 0.0,
      .ymax = 2.0};

  if (parallel.boss) {
    fputc('\n', g_out);
    if (use_fortran_kernels) {
      fputs("Fortran kernels were requested but they are not available\n", g_out);
      use_fortran_kernels = false;
      use_C_kernels = true;
      fputs("Using C Kernels\n", g_out);
    } else if (use_C_kernels) {
      fputs("Using C Kernels\n", g_out);
    } else if (use_OA_kernels) {
      fputs("Using OpenACC Kernels\n", g_out);
    }

    fputs("\nInput read finished.\n", g_out);
  }

  // If a state boundary falls exactly on a cell boundary then round off can
  // cause the state to be put one cell further that expected. This is compiler-
  // system dependent. To avoid this, a state boundary is reduced/increased by a 100th
  // of a cell width so it lies well with in the intended cell.
  // Because a cell is either full or empty of a specified state, this small
  // modification to the state extents does not change the answers.
  dx = (grid.xmax - grid.xmin) / (float)grid.x_cells;
  dy = (grid.ymax - grid.ymin) / (float)grid.y_cells;
  for (int i = 1; i < number_of_states; i++) {
    states[i].xmin = states[i].xmin + (dx / 100.0);
    states[i].ymin = states[i].ymin + (dy / 100.0);
    states[i].xmax = states[i].xmax - (dx / 100.0);
    states[i].ymax = states[i].ymax - (dy / 100.0);
  }
}

/**
 * @brief Main set up routine
 * @details Invokes the mesh decomposer and sets up chunk connectivity. It then allocates the communication buffers and
 * call the chunk initialisation and generation routines. It calls the equation of state to calculate initial pressure
 * before priming the halo cells and writing an initial field summary.
 */
void start() {
  int c, tile;

  int x_cells, y_cells;
  int right, left, top, bottom;

  int fields[NUM_FIELDS];

  bool profiler_off;

  if (parallel.boss) {
    fputs("\nSetting up initial geometry\n", g_out);
  }

  time_val = 0.0;
  step = 0;
  dtold = dtinit;
  dt = dtinit;

  number_of_chunks = clover_get_num_chunks();
  clover_decompose(grid.x_cells, grid.y_cells, &left, &right, &bottom, &top);

  // Create the chunks
  chunk.task = parallel.task;

  x_cells = right - left + 1;
  y_cells = top - bottom + 1;

  chunk.left = left;
  chunk.bottom = bottom;
  chunk.right = right;
  chunk.top = top;
  chunk.left_boundary = 1;
  chunk.bottom_boundary = 1;
  chunk.right_boundary = grid.x_cells;
  chunk.top_boundary = grid.y_cells;
  chunk.x_min = 1;
  chunk.y_min = 1;
  chunk.x_max = x_cells;
  chunk.y_max = y_cells;

  // Create the tiles
  chunk.tiles = malloc(tiles_per_chunk * sizeof(tile_type));
  clover_tile_decompose(x_cells, y_cells);

  build_field();

  if (parallel.boss)
    fputs("\nGenerating chunks\n", g_out);

  for (int tile = 0; tile < tiles_per_chunk; tile++) {
    initialise_chunk(tile);
    generate_chunk(tile);
  }

  advect_x = true;

  // Do no profile the start up costs otherwise the total times will not add up
  // at the end
  profiler_off = profiler_on;
  profiler_on = false;

  for (int tile = 0; tile < tiles_per_chunk; tile++)
    ideal_gas(tile, false);

  memset(fields, 0, sizeof(fields));
  fields[FIELD_DENSITY0] = 1;
  fields[FIELD_ENERGY0] = 1;
  fields[FIELD_PRESSURE] = 1;
  fields[FIELD_VISCOSITY] = 1;
  fields[FIELD_DENSITY1] = 1;
  fields[FIELD_ENERGY1] = 1;
  fields[FIELD_XVEL0] = 1;
  fields[FIELD_YVEL0] = 1;
  fields[FIELD_XVEL1] = 1;
  fields[FIELD_YVEL1] = 1;

  update_halo(fields, 2);

  if (parallel.boss)
    fputs("\nProblem initalised and generated\n", g_out);

  field_summary();

  if (visit_frequency != 0)
    visit();

  profiler_on = profiler_off;
}
