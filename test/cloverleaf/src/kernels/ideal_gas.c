/*Crown Copyright 2012 AWE.
 *
 * This file is part of CloverLeaf.
 *
 * CloverLeaf is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * CloverLeaf is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/**
 *  @brief C ideal gas kernel.
 *  @author Wayne Gaudin
 *  @details Calculates the pressure and sound speed for the mesh chunk using
 *  the ideal gas equation of state, with a fixed gamma of 1.4.
 */

#include <math.h>

#include "../types/definitions.h"
#include "ftocmacros.h"

void kernel_ideal_gas(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
    __attribute__((annotate("scalar(range(0,5))"))) double *density,
    __attribute__((annotate(RANGE_energy0))) double *energy,
    __attribute__((annotate(RANGE_pressure))) double *pressure,
    __attribute__((annotate(RANGE_soundspeed))) double *soundspeed
) {
  int j, k;
  __attribute__((annotate("scalar()"))) double sound_speed_squared;
  __attribute__((annotate("scalar()"))) double v;
  __attribute__((annotate("scalar()"))) double pressurebyenergy;
  __attribute__((annotate("scalar()"))) double pressurebyvolume;

  for (k = y_min; k <= y_max; k++) {
#pragma ivdep
    for (j = x_min; j <= x_max; j++) {
      v = 1.0 / density[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
      pressure[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] =
          (1.4 - 1.0) * density[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] *
          energy[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
      pressurebyenergy = (1.4 - 1.0) * density[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
      pressurebyvolume = -density[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] *
                         pressure[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
      sound_speed_squared =
          v * v * (pressure[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] * pressurebyenergy - pressurebyvolume);
      soundspeed[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] = sqrt(sound_speed_squared);
    }
  }
}
