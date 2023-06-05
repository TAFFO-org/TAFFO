/*Crown Copyright 2012 AWE.
 *
 * This file is part of CloverLeaf.
 *
 * CloverLeaf is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * CloverLeaf is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/**
 *  @brief C kernel to update the external halo cells in a chunk.
 *  @author Wayne Gaudin
 *  @details Updates halo cells for the required fields at the required depth
 *  for any halo cells that lie on an external boundary. The location and type
 *  of data governs how this is carried out. External boundaries are always
 *  reflective.
 */

#include "../types/definitions.h"
#include "data.h"
#include "ftocmacros.h"

void kernel_update_halo(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
    int chunk_neighbours[static 4],
    int tile_neighbours[static 4],
    __attribute__((annotate(RANGE_density0))) double *density0,
    __attribute__((annotate(RANGE_energy0))) double *energy0,
    __attribute__((annotate(RANGE_pressure))) double *pressure,
    __attribute__((annotate(RANGE_viscosity))) double *viscosity,
    __attribute__((annotate(RANGE_soundspeed))) double *soundspeed,
    __attribute__((annotate(RANGE_density1))) double *density1,
    __attribute__((annotate(RANGE_energy1))) double *energy1,
    __attribute__((annotate(RANGE_xvel0))) double *xvel0,
    __attribute__((annotate(RANGE_yvel0))) double *yvel0,
    __attribute__((annotate(RANGE_xvel1))) double *xvel1,
    __attribute__((annotate(RANGE_yvel1))) double *yvel1,
    __attribute__((annotate(RANGE_vol_flux_x))) double *vol_flux_x,
    __attribute__((annotate(RANGE_vol_flux_y))) double *vol_flux_y,
    __attribute__((annotate(RANGE_mass_flux_x))) double *mass_flux_x,
    __attribute__((annotate(RANGE_mass_flux_y))) double *mass_flux_y,
    int fields[static NUM_FIELDS],
    int depth
) {
  int j, k;

  /* Update values in external halo cells based on depth and fields requested */

  if (fields[FTNREF1D(FIELD_DENSITY0, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          density0[FTNREF2D(j, 1 - k, x_max + 4, x_min - 2, y_min - 2)] =
              density0[FTNREF2D(j, 0 + k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          density0[FTNREF2D(j, y_max + k, x_max + 4, x_min - 2, y_min - 2)] =
              density0[FTNREF2D(j, y_max + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          density0[FTNREF2D(1 - j, k, x_max + 4, x_min - 2, y_min - 2)] =
              density0[FTNREF2D(0 + j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          density0[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
              density0[FTNREF2D(x_max + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }
  }

  if (fields[FTNREF1D(FIELD_DENSITY1, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          density1[FTNREF2D(j, 1 - k, x_max + 4, x_min - 2, y_min - 2)] =
              density1[FTNREF2D(j, 0 + k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          density1[FTNREF2D(j, y_max + k, x_max + 4, x_min - 2, y_min - 2)] =
              density1[FTNREF2D(j, y_max + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          density1[FTNREF2D(1 - j, k, x_max + 4, x_min - 2, y_min - 2)] =
              density1[FTNREF2D(0 + j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          density1[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
              density1[FTNREF2D(x_max + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }
  }

  if (fields[FTNREF1D(FIELD_ENERGY0, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          energy0[FTNREF2D(j, 1 - k, x_max + 4, x_min - 2, y_min - 2)] =
              energy0[FTNREF2D(j, 0 + k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          energy0[FTNREF2D(j, y_max + k, x_max + 4, x_min - 2, y_min - 2)] =
              energy0[FTNREF2D(j, y_max + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          energy0[FTNREF2D(1 - j, k, x_max + 4, x_min - 2, y_min - 2)] =
              energy0[FTNREF2D(0 + j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          energy0[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
              energy0[FTNREF2D(x_max + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }
  }

  if (fields[FTNREF1D(FIELD_ENERGY1, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          energy1[FTNREF2D(j, 1 - k, x_max + 4, x_min - 2, y_min - 2)] =
              energy1[FTNREF2D(j, 0 + k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          energy1[FTNREF2D(j, y_max + k, x_max + 4, x_min - 2, y_min - 2)] =
              energy1[FTNREF2D(j, y_max + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          energy1[FTNREF2D(1 - j, k, x_max + 4, x_min - 2, y_min - 2)] =
              energy1[FTNREF2D(0 + j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          energy1[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
              energy1[FTNREF2D(x_max + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }
  }

  if (fields[FTNREF1D(FIELD_PRESSURE, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          pressure[FTNREF2D(j, 1 - k, x_max + 4, x_min - 2, y_min - 2)] =
              pressure[FTNREF2D(j, 0 + k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          pressure[FTNREF2D(j, y_max + k, x_max + 4, x_min - 2, y_min - 2)] =
              pressure[FTNREF2D(j, y_max + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          pressure[FTNREF2D(1 - j, k, x_max + 4, x_min - 2, y_min - 2)] =
              pressure[FTNREF2D(0 + j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          pressure[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
              pressure[FTNREF2D(x_max + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VISCOSITY, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          viscosity[FTNREF2D(j, 1 - k, x_max + 4, x_min - 2, y_min - 2)] =
              viscosity[FTNREF2D(j, 0 + k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          viscosity[FTNREF2D(j, y_max + k, x_max + 4, x_min - 2, y_min - 2)] =
              viscosity[FTNREF2D(j, y_max + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          viscosity[FTNREF2D(1 - j, k, x_max + 4, x_min - 2, y_min - 2)] =
              viscosity[FTNREF2D(0 + j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          viscosity[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
              viscosity[FTNREF2D(x_max + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }
  }

  if (fields[FTNREF1D(FIELD_SOUNDSPEED, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          soundspeed[FTNREF2D(j, 1 - k, x_max + 4, x_min - 2, y_min - 2)] =
              soundspeed[FTNREF2D(j, 0 + k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          soundspeed[FTNREF2D(j, y_max + k, x_max + 4, x_min - 2, y_min - 2)] =
              soundspeed[FTNREF2D(j, y_max + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          soundspeed[FTNREF2D(1 - j, k, x_max + 4, x_min - 2, y_min - 2)] =
              soundspeed[FTNREF2D(0 + j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          soundspeed[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
              soundspeed[FTNREF2D(x_max + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }
  }
  if (fields[FTNREF1D(FIELD_XVEL0, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          xvel0[FTNREF2D(j, 1 - k, x_max + 5, x_min - 2, y_min - 2)] =
              xvel0[FTNREF2D(j, 1 + k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          xvel0[FTNREF2D(j, y_max + 1 + k, x_max + 5, x_min - 2, y_min - 2)] =
              xvel0[FTNREF2D(j, y_max + 1 - k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          xvel0[FTNREF2D(1 - j, k, x_max + 5, x_min - 2, y_min - 2)] =
              -xvel0[FTNREF2D(1 + j, k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          xvel0[FTNREF2D(x_max + 1 + j, k, x_max + 5, x_min - 2, y_min - 2)] =
              -xvel0[FTNREF2D(x_max + 1 - j, k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }
  }

  if (fields[FTNREF1D(FIELD_XVEL1, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          xvel1[FTNREF2D(j, 1 - k, x_max + 5, x_min - 2, y_min - 2)] =
              xvel1[FTNREF2D(j, 1 + k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          xvel1[FTNREF2D(j, y_max + 1 + k, x_max + 5, x_min - 2, y_min - 2)] =
              xvel1[FTNREF2D(j, y_max + 1 - k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          xvel1[FTNREF2D(1 - j, k, x_max + 5, x_min - 2, y_min - 2)] =
              -xvel1[FTNREF2D(1 + j, k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          xvel1[FTNREF2D(x_max + 1 + j, k, x_max + 5, x_min - 2, y_min - 2)] =
              -xvel1[FTNREF2D(x_max + 1 - j, k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }
  }

  if (fields[FTNREF1D(FIELD_YVEL0, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          yvel0[FTNREF2D(j, 1 - k, x_max + 5, x_min - 2, y_min - 2)] =
              -yvel0[FTNREF2D(j, 1 + k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          yvel0[FTNREF2D(j, y_max + 1 + k, x_max + 5, x_min - 2, y_min - 2)] =
              -yvel0[FTNREF2D(j, y_max + 1 - k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          yvel0[FTNREF2D(1 - j, k, x_max + 5, x_min - 2, y_min - 2)] =
              yvel0[FTNREF2D(1 + j, k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          yvel0[FTNREF2D(x_max + 1 + j, k, x_max + 5, x_min - 2, y_min - 2)] =
              yvel0[FTNREF2D(x_max + 1 - j, k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }
  }

  if (fields[FTNREF1D(FIELD_YVEL1, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          yvel1[FTNREF2D(j, 1 - k, x_max + 5, x_min - 2, y_min - 2)] =
              -yvel1[FTNREF2D(j, 1 + k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          yvel1[FTNREF2D(j, y_max + 1 + k, x_max + 5, x_min - 2, y_min - 2)] =
              -yvel1[FTNREF2D(j, y_max + 1 - k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          yvel1[FTNREF2D(1 - j, k, x_max + 5, x_min - 2, y_min - 2)] =
              yvel1[FTNREF2D(1 + j, k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          yvel1[FTNREF2D(x_max + 1 + j, k, x_max + 5, x_min - 2, y_min - 2)] =
              yvel1[FTNREF2D(x_max + 1 - j, k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VOL_FLUX_X, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          vol_flux_x[FTNREF2D(j, 1 - k, x_max + 5, x_min - 2, y_min - 2)] =
              vol_flux_x[FTNREF2D(j, 1 + k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          vol_flux_x[FTNREF2D(j, y_max + k, x_max + 5, x_min - 2, y_min - 2)] =
              vol_flux_x[FTNREF2D(j, y_max - k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          vol_flux_x[FTNREF2D(1 - j, k, x_max + 5, x_min - 2, y_min - 2)] =
              -vol_flux_x[FTNREF2D(1 + j, k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }
    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          vol_flux_x[FTNREF2D(x_max + 1 + j, k, x_max + 5, x_min - 2, y_min - 2)] =
              -vol_flux_x[FTNREF2D(x_max + 1 - j, k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }
  }

  if (fields[FTNREF1D(FIELD_MASS_FLUX_X, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          mass_flux_x[FTNREF2D(j, 1 - k, x_max + 5, x_min - 2, y_min - 2)] =
              mass_flux_x[FTNREF2D(j, 1 + k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          mass_flux_x[FTNREF2D(j, y_max + k, x_max + 5, x_min - 2, y_min - 2)] =
              mass_flux_x[FTNREF2D(j, y_max - k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          mass_flux_x[FTNREF2D(1 - j, k, x_max + 5, x_min - 2, y_min - 2)] =
              -mass_flux_x[FTNREF2D(1 + j, k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          mass_flux_x[FTNREF2D(x_max + 1 + j, k, x_max + 5, x_min - 2, y_min - 2)] =
              -mass_flux_x[FTNREF2D(x_max + 1 - j, k, x_max + 5, x_min - 2, y_min - 2)];
        }
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VOL_FLUX_Y, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          vol_flux_y[FTNREF2D(j, 1 - k, x_max + 4, x_min - 2, y_min - 2)] =
              -vol_flux_y[FTNREF2D(j, 1 + k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          vol_flux_y[FTNREF2D(j, y_max + k + 1, x_max + 4, x_min - 2, y_min - 2)] =
              -vol_flux_y[FTNREF2D(j, y_max + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          vol_flux_y[FTNREF2D(1 - j, k, x_max + 4, x_min - 2, y_min - 2)] =
              vol_flux_y[FTNREF2D(1 + j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          vol_flux_y[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
              vol_flux_y[FTNREF2D(x_max - j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }
  }

  if (fields[FTNREF1D(FIELD_MASS_FLUX_Y, 1)] == 1) {
    if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          mass_flux_y[FTNREF2D(j, 1 - k, x_max + 4, x_min - 2, y_min - 2)] =
              -mass_flux_y[FTNREF2D(j, 1 + k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (k = 1; k <= depth; k++) {
          mass_flux_y[FTNREF2D(j, y_max + k + 1, x_max + 4, x_min - 2, y_min - 2)] =
              -mass_flux_y[FTNREF2D(j, y_max + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          mass_flux_y[FTNREF2D(1 - j, k, x_max + 4, x_min - 2, y_min - 2)] =
              mass_flux_y[FTNREF2D(1 + j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE &&
        tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {
      for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
        for (j = 1; j <= depth; j++) {
          mass_flux_y[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
              mass_flux_y[FTNREF2D(x_max - j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }
  }
}
