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
 *  @author Niccolò Betto, Wayne Gaudin
 *  @details Updates halo cells for the required fields at the required depth
 *  for any halo cells that lie on an external boundary. The location and type
 *  of data governs how this is carried out. External boundaries are always
 *  reflective.
 */

#include "../types/definitions.h"
#include "data.h"
#include "ftocmacros.h"

// clang-format on

void kernel_update_tile_halo_l(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
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
    int left_xmin,
    int left_xmax,
    int left_ymin,
    int left_ymax,
    __attribute__((annotate(RANGE_density0))) double *left_density0,
    __attribute__((annotate(RANGE_energy0))) double *left_energy0,
    __attribute__((annotate(RANGE_pressure))) double *left_pressure,
    __attribute__((annotate(RANGE_viscosity))) double *left_viscosity,
    __attribute__((annotate(RANGE_soundspeed))) double *left_soundspeed,
    __attribute__((annotate(RANGE_density1))) double *left_density1,
    __attribute__((annotate(RANGE_energy1))) double *left_energy1,
    __attribute__((annotate(RANGE_xvel0))) double *left_xvel0,
    __attribute__((annotate(RANGE_yvel0))) double *left_yvel0,
    __attribute__((annotate(RANGE_xvel1))) double *left_xvel1,
    __attribute__((annotate(RANGE_yvel1))) double *left_yvel1,
    __attribute__((annotate(RANGE_vol_flux_x))) double *left_vol_flux_x,
    __attribute__((annotate(RANGE_vol_flux_y))) double *left_vol_flux_y,
    __attribute__((annotate(RANGE_mass_flux_x))) double *left_mass_flux_x,
    __attribute__((annotate(RANGE_mass_flux_y))) double *left_mass_flux_y,
    int fields[static NUM_FIELDS],
    int depth
) {
  int j, k;

  if (fields[FTNREF1D(FTNREF1D(FIELD_DENSITY0, 1), 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        density0[FTNREF2D(j, x_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            left_density0[FTNREF2D(left_xmax + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_DENSITY1, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        density1[FTNREF2D(j, x_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            left_density1[FTNREF2D(left_xmax + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_ENERGY0, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        energy0[FTNREF2D(j, x_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            left_energy0[FTNREF2D(left_xmax + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_ENERGY1, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        energy1[FTNREF2D(j, x_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            left_energy1[FTNREF2D(left_xmax + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_PRESSURE, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        pressure[FTNREF2D(j, x_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            left_pressure[FTNREF2D(left_xmax + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VISCOSITY, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        viscosity[FTNREF2D(j, x_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            left_viscosity[FTNREF2D(left_xmax + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_SOUNDSPEED, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        soundspeed[FTNREF2D(j, x_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            left_soundspeed[FTNREF2D(left_xmax + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_XVEL0, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
      for (j = 0; j <= depth; j++) {
        xvel0[FTNREF2D(j, x_min - k, x_max + 5, x_min - 2, y_min - 2)] =
            left_xvel0[FTNREF2D(left_xmax + 1 - j, k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_XVEL1, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
      for (j = 0; j <= depth; j++) {
        xvel1[FTNREF2D(j, x_min - k, x_max + 5, x_min - 2, y_min - 2)] =
            left_xvel1[FTNREF2D(left_xmax + 1 - j, k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_YVEL0, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
      for (j = 0; j <= depth; j++) {
        yvel0[FTNREF2D(j, x_min - k, x_max + 5, x_min - 2, y_min - 2)] =
            left_yvel0[FTNREF2D(left_xmax + 1 - j, k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_YVEL1, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
      for (j = 0; j <= depth; j++) {
        yvel1[FTNREF2D(j, x_min - k, x_max + 5, x_min - 2, y_min - 2)] =
            left_yvel1[FTNREF2D(left_xmax + 1 - j, k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VOL_FLUX_X, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        vol_flux_x[FTNREF2D(j, x_min - k, x_max + 5, x_min - 2, y_min - 2)] =
            left_vol_flux_x[FTNREF2D(left_xmax + 1 - j, k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_MASS_FLUX_X, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        mass_flux_x[FTNREF2D(j, x_min - k, x_max + 5, x_min - 2, y_min - 2)] =
            left_mass_flux_x[FTNREF2D(left_xmax + 1 - j, k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VOL_FLUX_Y, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
      for (j = 0; j <= depth; j++) {
        vol_flux_y[FTNREF2D(j, x_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            left_vol_flux_y[FTNREF2D(left_xmax + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_MASS_FLUX_Y, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
      for (j = 0; j <= depth; j++) {
        mass_flux_y[FTNREF2D(j, x_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            left_mass_flux_y[FTNREF2D(left_xmax + 1 - j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }
}

void kernel_update_tile_halo_r(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
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
    int right_xmin,
    int right_xmax,
    int right_ymin,
    int right_ymax,
    __attribute__((annotate(RANGE_density0))) double *right_density0,
    __attribute__((annotate(RANGE_energy0))) double *right_energy0,
    __attribute__((annotate(RANGE_pressure))) double *right_pressure,
    __attribute__((annotate(RANGE_viscosity))) double *right_viscosity,
    __attribute__((annotate(RANGE_soundspeed))) double *right_soundspeed,
    __attribute__((annotate(RANGE_density1))) double *right_density1,
    __attribute__((annotate(RANGE_energy1))) double *right_energy1,
    __attribute__((annotate(RANGE_xvel0))) double *right_xvel0,
    __attribute__((annotate(RANGE_yvel0))) double *right_yvel0,
    __attribute__((annotate(RANGE_xvel1))) double *right_xvel1,
    __attribute__((annotate(RANGE_yvel1))) double *right_yvel1,
    __attribute__((annotate(RANGE_vol_flux_x))) double *right_vol_flux_x,
    __attribute__((annotate(RANGE_vol_flux_y))) double *right_vol_flux_y,
    __attribute__((annotate(RANGE_mass_flux_x))) double *right_mass_flux_x,
    __attribute__((annotate(RANGE_mass_flux_y))) double *right_mass_flux_y,
    int fields[static NUM_FIELDS],
    int depth
) {
  int j, k;

  if (fields[FTNREF1D(FIELD_DENSITY0, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        density0[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
            right_density0[FTNREF2D(right_xmin - 1 + j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_DENSITY1, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        density1[FTNREF2D(k, x_max + j, x_max + 4, x_min - 2, y_min - 2)] =
            right_density1[FTNREF2D(right_xmin - 1 + j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_ENERGY0, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        energy0[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
            right_energy0[FTNREF2D(right_xmin - 1 + j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_ENERGY1, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        energy1[FTNREF2D(k, x_max + j, x_max + 4, x_min - 2, y_min - 2)] =
            right_energy1[FTNREF2D(right_xmin - 1 + j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_PRESSURE, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        pressure[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
            right_pressure[FTNREF2D(right_xmin - 1 + j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VISCOSITY, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        viscosity[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
            right_viscosity[FTNREF2D(right_xmin - 1 + j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_SOUNDSPEED, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        soundspeed[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
            right_soundspeed[FTNREF2D(right_xmin - 1 + j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_XVEL0, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
      for (j = 0; j <= depth; j++) {
        xvel0[FTNREF2D(x_max + 1 + j, k, x_max + 5, x_min - 2, y_min - 2)] =
            right_xvel0[FTNREF2D(right_xmin + 1 - 1 + j, k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_XVEL1, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
      for (j = 0; j <= depth; j++) {
        xvel1[FTNREF2D(x_max + 1 + j, k, x_max + 5, x_min - 2, y_min - 2)] =
            right_xvel1[FTNREF2D(right_xmin + 1 - 1 + j, k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_YVEL0, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
      for (j = 0; j <= depth; j++) {
        yvel0[FTNREF2D(x_max + 1 + j, k, x_max + 5, x_min - 2, y_min - 2)] =
            right_yvel0[FTNREF2D(right_xmin + 1 - 1 + j, k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_YVEL1, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
      for (j = 0; j <= depth; j++) {
        yvel1[FTNREF2D(x_max + 1 + j, k, x_max + 5, x_min - 2, y_min - 2)] =
            right_yvel1[FTNREF2D(right_xmin + 1 - 1 + j, k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VOL_FLUX_X, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        vol_flux_x[FTNREF2D(x_max + 1 + j, k, x_max + 5, x_min - 2, y_min - 2)] =
            right_vol_flux_x[FTNREF2D(right_xmin + 1 - 1 + j, k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_MASS_FLUX_X, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + depth; k++) {
      for (j = 0; j <= depth; j++) {
        mass_flux_x[FTNREF2D(x_max + 1 + j, k, x_max + 5, x_min - 2, y_min - 2)] =
            right_mass_flux_x[FTNREF2D(right_xmin + 1 - 1 + j, k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VOL_FLUX_Y, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
      for (j = 0; j <= depth; j++) {
        vol_flux_y[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
            right_vol_flux_y[FTNREF2D(right_xmin - 1 + j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_MASS_FLUX_Y, 1)] == 1) {
    for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
      for (j = 0; j <= depth; j++) {
        mass_flux_y[FTNREF2D(x_max + j, k, x_max + 4, x_min - 2, y_min - 2)] =
            right_mass_flux_y[FTNREF2D(right_xmin - 1 + j, k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }
}

void kernel_update_tile_halo_t(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
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
    int top_xmin,
    int top_xmax,
    int top_ymin,
    int top_ymax,
    __attribute__((annotate(RANGE_density0))) double *top_density0,
    __attribute__((annotate(RANGE_energy0))) double *top_energy0,
    __attribute__((annotate(RANGE_pressure))) double *top_pressure,
    __attribute__((annotate(RANGE_viscosity))) double *top_viscosity,
    __attribute__((annotate(RANGE_soundspeed))) double *top_soundspeed,
    __attribute__((annotate(RANGE_density1))) double *top_density1,
    __attribute__((annotate(RANGE_energy1))) double *top_energy1,
    __attribute__((annotate(RANGE_xvel0))) double *top_xvel0,
    __attribute__((annotate(RANGE_yvel0))) double *top_yvel0,
    __attribute__((annotate(RANGE_xvel1))) double *top_xvel1,
    __attribute__((annotate(RANGE_yvel1))) double *top_yvel1,
    __attribute__((annotate(RANGE_vol_flux_x))) double *top_vol_flux_x,
    __attribute__((annotate(RANGE_vol_flux_y))) double *top_vol_flux_y,
    __attribute__((annotate(RANGE_mass_flux_x))) double *top_mass_flux_x,
    __attribute__((annotate(RANGE_mass_flux_y))) double *top_mass_flux_y,
    int fields[static NUM_FIELDS],
    int depth
) {
  int j, k;

  if (fields[FTNREF1D(FIELD_DENSITY0, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        density0[FTNREF2D(j, y_max + k, x_max + 4, x_min - 2, y_min - 2)] =
            top_density0[FTNREF2D(j, top_ymin - 1 + k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_DENSITY1, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        density1[FTNREF2D(j, y_max + k, x_max + 4, x_min - 2, y_min - 2)] =
            top_density1[FTNREF2D(j, top_ymin - 1 + k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_ENERGY0, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        energy0[FTNREF2D(j, y_max + k, x_max + 4, x_min - 2, y_min - 2)] =
            top_energy0[FTNREF2D(j, top_ymin - 1 + k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_ENERGY1, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        energy1[FTNREF2D(j, y_max + k, x_max + 4, x_min - 2, y_min - 2)] =
            top_energy1[FTNREF2D(j, top_ymin - 1 + k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_PRESSURE, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        pressure[FTNREF2D(j, y_max + k, x_max + 4, x_min - 2, y_min - 2)] =
            top_pressure[FTNREF2D(j, top_ymin - 1 + k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VISCOSITY, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        viscosity[FTNREF2D(j, y_max + k, x_max + 4, x_min - 2, y_min - 2)] =
            top_viscosity[FTNREF2D(j, top_ymin - 1 + k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_SOUNDSPEED, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        soundspeed[FTNREF2D(j, y_max + k, x_max + 4, x_min - 2, y_min - 2)] =
            top_soundspeed[FTNREF2D(j, top_ymin - 1 + k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_XVEL0, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
        xvel0[FTNREF2D(j, y_max + 1 + k, x_max + 5, x_min - 2, y_min - 2)] =
            top_xvel0[FTNREF2D(j, top_ymin + 1 - 1 + k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_XVEL1, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
        xvel1[FTNREF2D(j, y_max + 1 + k, x_max + 5, x_min - 2, y_min - 2)] =
            top_xvel1[FTNREF2D(j, top_ymin + 1 - 1 + k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_YVEL0, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
        yvel0[FTNREF2D(j, y_max + 1 + k, x_max + 5, x_min - 2, y_min - 2)] =
            top_yvel0[FTNREF2D(j, top_ymin + 1 - 1 + k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_YVEL1, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
        yvel1[FTNREF2D(j, y_max + 1 + k, x_max + 5, x_min - 2, y_min - 2)] =
            top_yvel1[FTNREF2D(j, top_ymin + 1 - 1 + k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VOL_FLUX_X, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
        vol_flux_x[FTNREF2D(j, y_max + k, x_max + 5, x_min - 2, y_min - 2)] =
            top_vol_flux_x[FTNREF2D(j, top_ymin - 1 + k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_MASS_FLUX_X, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
        mass_flux_x[FTNREF2D(j, y_max + k, x_max + 5, x_min - 2, y_min - 2)] =
            top_mass_flux_x[FTNREF2D(j, top_ymin - 1 + k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VOL_FLUX_Y, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        vol_flux_y[FTNREF2D(j, y_max + 1 + k, x_max + 4, x_min - 2, y_min - 2)] =
            top_vol_flux_y[FTNREF2D(j, top_ymin + 1 - 1 + k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_MASS_FLUX_Y, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        mass_flux_y[FTNREF2D(j, y_max + 1 + k, x_max + 4, x_min - 2, y_min - 2)] =
            top_mass_flux_y[FTNREF2D(j, top_ymin + 1 - 1 + k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }
}

void kernel_update_tile_halo_b(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
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
    int bottom_xmin,
    int bottom_xmax,
    int bottom_ymin,
    int bottom_ymax,
    __attribute__((annotate(RANGE_density0))) double *bottom_density0,
    __attribute__((annotate(RANGE_energy0))) double *bottom_energy0,
    __attribute__((annotate(RANGE_pressure))) double *bottom_pressure,
    __attribute__((annotate(RANGE_viscosity))) double *bottom_viscosity,
    __attribute__((annotate(RANGE_soundspeed))) double *bottom_soundspeed,
    __attribute__((annotate(RANGE_density1))) double *bottom_density1,
    __attribute__((annotate(RANGE_energy1))) double *bottom_energy1,
    __attribute__((annotate(RANGE_xvel0))) double *bottom_xvel0,
    __attribute__((annotate(RANGE_yvel0))) double *bottom_yvel0,
    __attribute__((annotate(RANGE_xvel1))) double *bottom_xvel1,
    __attribute__((annotate(RANGE_yvel1))) double *bottom_yvel1,
    __attribute__((annotate(RANGE_vol_flux_x))) double *bottom_vol_flux_x,
    __attribute__((annotate(RANGE_vol_flux_y))) double *bottom_vol_flux_y,
    __attribute__((annotate(RANGE_mass_flux_x))) double *bottom_mass_flux_x,
    __attribute__((annotate(RANGE_mass_flux_y))) double *bottom_mass_flux_y,
    int fields[static NUM_FIELDS],
    int depth
) {
  int j, k;

  if (fields[FTNREF1D(FIELD_DENSITY0, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        density0[FTNREF2D(j, y_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            bottom_density0[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_DENSITY1, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        density1[FTNREF2D(j, y_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            bottom_density1[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_ENERGY0, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        energy0[FTNREF2D(j, y_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            bottom_energy0[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_ENERGY1, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        energy1[FTNREF2D(j, y_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            bottom_energy1[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_PRESSURE, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        pressure[FTNREF2D(j, y_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            bottom_pressure[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VISCOSITY, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        viscosity[FTNREF2D(j, y_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            bottom_viscosity[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_SOUNDSPEED, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        soundspeed[FTNREF2D(j, y_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            bottom_soundspeed[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_XVEL0, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
        xvel0[FTNREF2D(j, y_min - k, x_max + 5, x_min - 2, y_min - 2)] =
            bottom_xvel0[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_XVEL1, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
        xvel1[FTNREF2D(j, y_min - k, x_max + 5, x_min - 2, y_min - 2)] =
            bottom_xvel1[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_YVEL0, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
        yvel0[FTNREF2D(j, y_min - k, x_max + 5, x_min - 2, y_min - 2)] =
            bottom_yvel0[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_YVEL1, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
        yvel1[FTNREF2D(j, y_min - k, x_max + 5, x_min - 2, y_min - 2)] =
            bottom_yvel1[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VOL_FLUX_X, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
        vol_flux_x[FTNREF2D(j, y_min - k, x_max + 5, x_min - 2, y_min - 2)] =
            bottom_vol_flux_x[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_MASS_FLUX_X, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
        mass_flux_x[FTNREF2D(j, y_min - k, x_max + 5, x_min - 2, y_min - 2)] =
            bottom_mass_flux_x[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_VOL_FLUX_Y, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        vol_flux_y[FTNREF2D(j, y_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            bottom_vol_flux_y[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }

  if (fields[FTNREF1D(FIELD_MASS_FLUX_Y, 1)] == 1) {
    for (k = 0; k <= depth; k++) {
      for (j = x_min - depth; j <= x_max + depth; j++) {
        mass_flux_y[FTNREF2D(j, y_min - k, x_max + 4, x_min - 2, y_min - 2)] =
            bottom_mass_flux_y[FTNREF2D(j, bottom_ymax + 1 - k, x_max + 4, x_min - 2, y_min - 2)];
      }
    }
  }
}
