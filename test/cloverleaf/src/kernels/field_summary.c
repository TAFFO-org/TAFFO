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
 *  @brief C field summary kernel
 *  @author Wayne Gaudin
 *  @details The total mass, internal energy, kinetic energy and volume weighted
 *  pressure for the chunk is calculated.
 */

#include "../types/definitions.h"
#include "ftocmacros.h"

void kernel_field_summary(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
    __attribute__((annotate(RANGE_volume))) double *volume,
    __attribute__((annotate(RANGE_density0))) double *density0,
    __attribute__((annotate(RANGE_energy0))) double *energy0,
    __attribute__((annotate(RANGE_pressure))) double *pressure,
    __attribute__((annotate(RANGE_xvel0))) double *xvel0,
    __attribute__((annotate(RANGE_yvel0))) double *yvel0,
    __attribute__((annotate(RANGE_vol))) double *p_vol,
    __attribute__((annotate(RANGE_mass))) double *p_mass,
    __attribute__((annotate(RANGE_ie))) double *p_ie,
    __attribute__((annotate(RANGE_ke))) double *p_ke,
    __attribute__((annotate(RANGE_press))) double *p_press
) {
  int j, k, jv, kv;

  __attribute__((annotate(RANGE_vol))) double vol = *p_vol;
  __attribute__((annotate(RANGE_mass))) double mass = *p_mass;
  __attribute__((annotate(RANGE_ie))) double ie = *p_ie;
  __attribute__((annotate(RANGE_ke))) double ke = *p_ke;
  __attribute__((annotate(RANGE_press))) double press = *p_press;

  __attribute__((annotate(RANGE_ke))) double vsqrd;
  __attribute__((annotate(RANGE_vol))) double cell_vol;
  __attribute__((annotate(RANGE_mass))) double cell_mass;

  vol = 0.0;
  mass = 0.0;
  ie = 0.0;
  ke = 0.0;
  press = 0.0;

  for (k = y_min; k <= y_max; k++) {
#pragma ivdep
    for (j = x_min; j <= x_max; j++) {
      vsqrd = 0.0;
      for (kv = k; kv <= k + 1; kv++) {
        for (jv = j; jv <= j + 1; jv++) {
          vsqrd = vsqrd + 0.25 * (xvel0[FTNREF2D(jv, kv, x_max + 5, x_min - 2, y_min - 2)] *
                                      xvel0[FTNREF2D(jv, kv, x_max + 5, x_min - 2, y_min - 2)] +
                                  yvel0[FTNREF2D(jv, kv, x_max + 5, x_min - 2, y_min - 2)] *
                                      yvel0[FTNREF2D(jv, kv, x_max + 5, x_min - 2, y_min - 2)]);
        }
      }
      cell_vol = volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
      cell_mass = cell_vol * density0[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
      vol = vol + cell_vol;
      mass = mass + cell_mass;
      ie = ie + cell_mass * energy0[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
      ke = ke + cell_mass * 0.5 * vsqrd;
      press = press + cell_vol * pressure[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
    }
  }

  *p_vol = vol;
  *p_mass = mass;
  *p_ie = ie;
  *p_ke = ke;
  *p_press = press;
}
