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
 *  @brief C acceleration kernel
 *  @author Wayne Gaudin
 *  @details The pressure and viscosity gradients are used to update the
 *  velocity field.
 */

#include "ftocmacros.h"

void kernel_accelerate(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
    double dt,
    double *xarea,
    double *yarea,
    double *volume,
    double *density0,
    double *pressure,
    double *viscosity,
    double *xvel0,
    double *yvel0,
    double *xvel1,
    double *yvel1
) {
  int j, k, err;
  double nodal_mass;
  double stepby_mass_s;

  for (k = y_min; k <= y_max + 1; k++) {
#pragma ivdep
    for (j = x_min; j <= x_max + 1; j++) {
      nodal_mass = (density0[FTNREF2D(j - 1, k - 1, x_max + 4, x_min - 2, y_min - 2)] *
                        volume[FTNREF2D(j - 1, k - 1, x_max + 4, x_min - 2, y_min - 2)] +
                    density0[FTNREF2D(j, k - 1, x_max + 4, x_min - 2, y_min - 2)] *
                        volume[FTNREF2D(j, k - 1, x_max + 4, x_min - 2, y_min - 2)] +
                    density0[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] *
                        volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] +
                    density0[FTNREF2D(j - 1, k, x_max + 4, x_min - 2, y_min - 2)] *
                        volume[FTNREF2D(j - 1, k, x_max + 4, x_min - 2, y_min - 2)]) *
                   0.25;
      stepby_mass_s = 0.5 * dt / nodal_mass;
      xvel1[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
          xvel0[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] -
          stepby_mass_s * (xarea[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] *
                               (pressure[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] -
                                pressure[FTNREF2D(j - 1, k, x_max + 4, x_min - 2, y_min - 2)]) +
                           xarea[FTNREF2D(j, k - 1, x_max + 5, x_min - 2, y_min - 2)] *
                               (pressure[FTNREF2D(j, k - 1, x_max + 4, x_min - 2, y_min - 2)] -
                                pressure[FTNREF2D(j - 1, k - 1, x_max + 4, x_min - 2, y_min - 2)]));

      yvel1[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
          yvel0[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] -
          stepby_mass_s * (yarea[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] *
                               (pressure[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] -
                                pressure[FTNREF2D(j, k - 1, x_max + 4, x_min - 2, y_min - 2)]) +
                           yarea[FTNREF2D(j - 1, k, x_max + 4, x_min - 2, y_min - 2)] *
                               (pressure[FTNREF2D(j - 1, k, x_max + 4, x_min - 2, y_min - 2)] -
                                pressure[FTNREF2D(j - 1, k - 1, x_max + 4, x_min - 2, y_min - 2)]));

      xvel1[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
          xvel1[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] -
          stepby_mass_s * (xarea[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] *
                               (viscosity[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] -
                                viscosity[FTNREF2D(j - 1, k, x_max + 4, x_min - 2, y_min - 2)]) +
                           xarea[FTNREF2D(j, k - 1, x_max + 5, x_min - 2, y_min - 2)] *
                               (viscosity[FTNREF2D(j, k - 1, x_max + 4, x_min - 2, y_min - 2)] -
                                viscosity[FTNREF2D(j - 1, k - 1, x_max + 4, x_min - 2, y_min - 2)]));

      yvel1[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
          yvel1[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] -
          stepby_mass_s * (yarea[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] *
                               (viscosity[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] -
                                viscosity[FTNREF2D(j, k - 1, x_max + 4, x_min - 2, y_min - 2)]) +
                           yarea[FTNREF2D(j - 1, k, x_max + 4, x_min - 2, y_min - 2)] *
                               (viscosity[FTNREF2D(j - 1, k, x_max + 4, x_min - 2, y_min - 2)] -
                                viscosity[FTNREF2D(j - 1, k - 1, x_max + 4, x_min - 2, y_min - 2)]));
    }
  }
}
