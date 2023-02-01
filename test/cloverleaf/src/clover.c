// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data.h"
#include "definitions.h"

void clover_init_comms() {
  const int rank = 0, size = 1;

  parallel.parallel = true;
  parallel.task = rank;

  if (rank == 0)
    parallel.boss = false;

  parallel.boss_task = 0;
  parallel.max_task = size;
}

void clover_finalize() {
}

void clover_abort() {
  clover_finalize();
  exit(1);
}

int clover_get_num_chunks() {
  return parallel.max_task;
}

void clover_decompose(int x_cells, int y_cells, int *left, int *right, int *bottom, int *top) {
  // This decomposes the mesh into a number of chunks.
  // The number of chunks may be a multiple of the number of mpi tasks
  // Doesn't always return the best split if there are few factors
  // All factors need to be stored and the best picked. But its ok for now

  int delta_x, delta_y;

  double mesh_ratio, factor_x, factor_y;
  int chunk_x, chunk_y, mod_x, mod_y;

  int cx, cy, cnk, add_x, add_y, add_x_prev, add_y_prev;

  // 2D decomposition of the mesh

  mesh_ratio = (double)x_cells / (double)y_cells;

  chunk_x = number_of_chunks;
  chunk_y = 1;

  bool split_found = false;  // Used to detect 1D decomposition

  for (int c = 1; c <= number_of_chunks; c++) {
    if (number_of_chunks % c == 0) {
      factor_x = number_of_chunks / (double)c;
      factor_y = c;
      // Compare the factor ratio with the mesh ratio
      if (factor_x / factor_y <= mesh_ratio) {
        chunk_y = c;
        chunk_x = number_of_chunks / c;
        split_found = true;
      }
    }
  }

  if (!split_found || chunk_y == number_of_chunks) {
    // Prime number or 1D decomp detected
    if (mesh_ratio >= 1.0) {
      chunk_x = number_of_chunks;
      chunk_y = 1;
    } else {
      chunk_x = 1;
      chunk_y = number_of_chunks;
    }
  }

  delta_x = x_cells / chunk_x;
  delta_y = y_cells / chunk_y;
  mod_x = x_cells % chunk_x;
  mod_y = y_cells % chunk_y;

  // Set up chunk mesh ranges and chunk connectivity

  add_x_prev = 0;
  add_y_prev = 0;
  cnk = 1;

  for (int cy = 1; cy <= chunk_y; cy++) {
    for (int cx = 1; cx <= chunk_x; cx++) {
      add_x = 0;
      add_y = 0;
      if (cx <= mod_x)
        add_x = 1;
      if (cy <= mod_y)
        add_y = 1;

      if (cnk == parallel.task + 1) {
        *left = (cx - 1) * delta_x + 1 + add_x_prev;
        *right = *left + delta_x - 1 + add_x;
        *bottom = (cy - 1) * delta_y + 1 + add_y_prev;
        *top = *bottom + delta_y - 1 + add_y;

        chunk.chunk_neighbours[CHUNK_LEFT] = chunk_x * (cy - 1) + cx - 1;
        chunk.chunk_neighbours[CHUNK_RIGHT] = chunk_x * (cy - 1) + cx + 1;
        chunk.chunk_neighbours[CHUNK_BOTTOM] = chunk_x * (cy - 2) + cx;
        chunk.chunk_neighbours[CHUNK_TOP] = chunk_x * cy + cx;

        if (cx == 1)
          chunk.chunk_neighbours[CHUNK_LEFT] = EXTERNAL_FACE;
        if (cx == chunk_x)
          chunk.chunk_neighbours[CHUNK_RIGHT] = EXTERNAL_FACE;
        if (cy == 1)
          chunk.chunk_neighbours[CHUNK_BOTTOM] = EXTERNAL_FACE;
        if (cy == chunk_y)
          chunk.chunk_neighbours[CHUNK_TOP] = EXTERNAL_FACE;
      }

      if (cx <= mod_x)
        add_x_prev++;
      cnk++;
    }
    add_x_prev = 0;
    if (cy <= mod_y)
      add_y_prev++;
  }

  if (parallel.boss) {
    fprintf(g_out, "\nMesh ratio of %4.16lf\n", mesh_ratio);
    fprintf(g_out, "Decomposing the mesh into %d by %d chunks\n", chunk_x, chunk_y);
    fprintf(g_out, "Decomposing the chunk with %d tiles\n", tiles_per_chunk);
  }
}

void clover_tile_decompose(int chunk_x_cells, int chunk_y_cells) {
  int chunk_mesh_ratio = (float)chunk_x_cells / (float)chunk_y_cells;

  int tile_x = tiles_per_chunk;
  int tile_y = 1;

  bool split_found = false;  // Used to detect 1D decomposition

  for (int t = 1; t <= tiles_per_chunk; t++) {
    if (tiles_per_chunk % t == 0) {
      int factor_x = tiles_per_chunk / (float)t;
      int factor_y = t;
      // Compare the factor ration with the mesh ratio
      if (factor_x / factor_y <= chunk_mesh_ratio) {
        tile_y = t;
        tile_x = tiles_per_chunk / t;
        split_found = true;
      }
    }
  }

  if (!split_found || tile_y == tiles_per_chunk) {
    // Prime number or 1D decomp detected
    if (chunk_mesh_ratio >= 1.0) {
      tile_x = tiles_per_chunk;
      tile_y = 1;
    } else {
      tile_x = 1;
      tile_y = tiles_per_chunk;
    }
  }

  int chunk_delta_x = chunk_x_cells / tile_x;
  int chunk_delta_y = chunk_y_cells / tile_y;
  int chunk_mod_x = chunk_x_cells % tile_x;
  int chunk_mod_y = chunk_y_cells % tile_y;

  int add_x_prev = 0;
  int add_y_prev = 0;
  int tile = 0;

  for (int ty = 1; ty <= tile_y; ty++) {
    for (int tx = 1; tx <= tile_x; tx++) {
      int add_x = 0;
      int add_y = 0;

      if (tx <= chunk_mod_x)
        add_x = 1;
      if (ty <= chunk_mod_y)
        add_y = 1;

      int left = chunk.left + (tx - 1) * chunk_delta_x + add_x_prev;
      int right = left + chunk_delta_x - 1 + add_x;
      int bottom = chunk.bottom + (ty - 1) * chunk_delta_y + add_y_prev;
      int top = bottom + chunk_delta_y - 1 + add_y;

      chunk.tiles[tile].tile_neighbours[TILE_LEFT] = tile_x * (ty - 1) + tx - 1;
      chunk.tiles[tile].tile_neighbours[TILE_RIGHT] = tile_x * (ty - 1) + tx + 1;
      chunk.tiles[tile].tile_neighbours[TILE_BOTTOM] = tile_x * (ty - 2) + tx;
      chunk.tiles[tile].tile_neighbours[TILE_TOP] = tile_x * ty + tx;

      // Initial set the external tile mast to 0 for each tile
      memset(chunk.tiles[tile].external_tile_mask, 0, sizeof(chunk.tiles[tile].external_tile_mask));

      if (tx == 1) {
        chunk.tiles[tile].tile_neighbours[TILE_LEFT] = EXTERNAL_TILE;
        chunk.tiles[tile].external_tile_mask[TILE_LEFT] = 1;
      }

      if (tx == tile_x) {
        chunk.tiles[tile].tile_neighbours[TILE_RIGHT] = EXTERNAL_TILE;
        chunk.tiles[tile].external_tile_mask[TILE_RIGHT] = 1;
      }

      if (ty == 1) {
        chunk.tiles[tile].tile_neighbours[TILE_BOTTOM] = EXTERNAL_TILE;
        chunk.tiles[tile].external_tile_mask[TILE_BOTTOM] = 1;
      }

      if (ty == tile_y) {
        chunk.tiles[tile].tile_neighbours[TILE_TOP] = EXTERNAL_TILE;
        chunk.tiles[tile].external_tile_mask[TILE_TOP] = 1;
      }

      if (tx <= chunk_mod_x)
        add_x_prev++;

      chunk.tiles[tile].t_xmin = 1;
      chunk.tiles[tile].t_xmax = right - left + 1;
      chunk.tiles[tile].t_ymin = 1;
      chunk.tiles[tile].t_ymax = top - bottom + 1;

      chunk.tiles[tile].t_left = left;
      chunk.tiles[tile].t_right = right;
      chunk.tiles[tile].t_top = top;
      chunk.tiles[tile].t_bottom = bottom;

      tile++;
    }

    add_x_prev = 0;
    if (ty <= chunk_mod_y)
      add_y_prev++;
  }
}
