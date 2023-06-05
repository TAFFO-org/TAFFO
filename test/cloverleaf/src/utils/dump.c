// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Niccolò Betto

/*
 * Utility functions to dump the program's arrays to file
 */

#include <errno.h>
#include <stdio.h>

#include "../definitions.h"

FILE *dump_file = NULL;

void dump_init() {
  errno = 0;
  dump_file = fopen("dump.txt", "w");
  if (errno != 0) {
    printf("Error opening dump.txt file, output will go to stdout.\n");
    dump_file = stdout;
  }
}

void dump_step_header(int step) {
  fprintf(dump_file, "============== Step: %d ==============\n", step);
}

#define DUMP_2D_DEF(array, xmax, xmin, ymax, ymin)                                       \
  void dump_##array(int tile) {                                                          \
    tile_type *cur_tile = &chunk.tiles[tile];                                            \
                                                                                         \
    fprintf(                                                                             \
        dump_file,                                                                       \
        "DUMP " #array " [%dx%d]\n",                                                     \
        (cur_tile->t_xmax + xmax) - (cur_tile->t_xmin + xmin),                           \
        (cur_tile->t_ymax + ymax) - (cur_tile->t_ymin + ymin)                            \
    );                                                                                   \
                                                                                         \
    int stride = (cur_tile->t_xmax + xmax) - (cur_tile->t_xmin + xmin) + 1;              \
    for (int k = 0; k <= (cur_tile->t_ymax + ymax) - (cur_tile->t_ymin + ymin); k++) {   \
      for (int j = 0; j <= (cur_tile->t_xmax + xmax) - (cur_tile->t_xmin + xmin); j++) { \
        fprintf(dump_file, "%.2f ", cur_tile->field.array[k * stride + j]);              \
      }                                                                                  \
      fprintf(dump_file, "\n");                                                          \
    }                                                                                    \
  }

#define DUMP_1DX_DEF(array, xmax, xmin)                                                \
  void dump_##array(int tile) {                                                        \
    tile_type *cur_tile = &chunk.tiles[tile];                                          \
                                                                                       \
    puts("DUMP " #array "\n");                                                         \
    for (int j = 0; j <= (cur_tile->t_xmax + xmax) - (cur_tile->t_xmin + xmin); j++) { \
      fprintf(dump_file, "%.2f ", cur_tile->field.array[j]);                           \
    }                                                                                  \
    fprintf(dump_file, "\n");                                                          \
  }

#define DUMP_1DY_DEF(array, ymax, ymin)                                                \
  void dump_##array(int tile) {                                                        \
    tile_type *cur_tile = &chunk.tiles[tile];                                          \
                                                                                       \
    puts("DUMP " #array "\n");                                                         \
    for (int j = 0; j <= (cur_tile->t_ymax + ymax) - (cur_tile->t_ymin + ymin); j++) { \
      fprintf(dump_file, "%.2f ", cur_tile->field.array[j]);                           \
    }                                                                                  \
    fprintf(dump_file, "\n");                                                          \
  }

#define ARRAY_2D_DEF(array, xmax, xmin, ymax, ymin) DUMP_2D_DEF(array, xmax, xmin, ymax, ymin)

#define ARRAY_1DX_DEF(array, xmax, xmin) DUMP_1DX_DEF(array, xmax, xmin)

#define ARRAY_1DY_DEF(array, ymax, ymin) DUMP_1DY_DEF(array, ymax, ymin)

ARRAY_2D_DEF(density0, 2, -2, 2, -2)
ARRAY_2D_DEF(density1, 2, -2, 2, -2)
ARRAY_2D_DEF(energy0, 2, -2, 2, -2)
ARRAY_2D_DEF(energy1, 2, -2, 2, -2)
ARRAY_2D_DEF(pressure, 2, -2, 2, -2)
ARRAY_2D_DEF(viscosity, 2, -2, 2, -2)
ARRAY_2D_DEF(soundspeed, 2, -2, 2, -2)
ARRAY_2D_DEF(volume, 2, -2, 2, -2)

ARRAY_2D_DEF(xvel0, 3, -2, 3, -2)
ARRAY_2D_DEF(xvel1, 3, -2, 3, -2)
ARRAY_2D_DEF(yvel0, 3, -2, 3, -2)
ARRAY_2D_DEF(yvel1, 3, -2, 3, -2)

ARRAY_2D_DEF(work_array1, 3, -2, 3, -2)
ARRAY_2D_DEF(work_array2, 3, -2, 3, -2)
ARRAY_2D_DEF(work_array3, 3, -2, 3, -2)
ARRAY_2D_DEF(work_array4, 3, -2, 3, -2)
ARRAY_2D_DEF(work_array5, 3, -2, 3, -2)
ARRAY_2D_DEF(work_array6, 3, -2, 3, -2)
ARRAY_2D_DEF(work_array7, 3, -2, 3, -2)

ARRAY_2D_DEF(vol_flux_x, 3, -2, 2, -2)
ARRAY_2D_DEF(mass_flux_x, 3, -2, 2, -2)
ARRAY_2D_DEF(xarea, 2, -2, 3, -2)

ARRAY_2D_DEF(vol_flux_y, 2, -2, 3, -2)
ARRAY_2D_DEF(mass_flux_y, 2, -2, 3, -2)
ARRAY_2D_DEF(yarea, 3, -2, 2, -2)

ARRAY_1DX_DEF(cellx, 2, -2)
ARRAY_1DX_DEF(celldx, 2, -2)

ARRAY_1DY_DEF(celly, 2, -2)
ARRAY_1DY_DEF(celldy, 2, -2)

ARRAY_1DX_DEF(vertexx, 3, -2)
ARRAY_1DX_DEF(vertexdx, 3, -2)

ARRAY_1DY_DEF(vertexy, 3, -2)
ARRAY_1DY_DEF(vertexdy, 3, -2)

#undef DUMP_2D_DEF
#undef DUMP_1DX_DEF
#undef DUMP_1DY_DEF
#undef ARRAY_2D_DEF
#undef ARRAY_1DX_DEF
#undef ARRAY_1DY_DEF
