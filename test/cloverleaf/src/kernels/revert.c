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
 *  @brief C revert kernel.
 *  @author Wayne Gaudin
 *  @details Takes the half step field data used in the predictor and reverts
 *  it to the start of step data, ready for the corrector.
 *  Note that this does not seem necessary in this proxy-app but should be
 *  left in to remain relevant to the full method.
 */

#include "../types/definitions.h"
#include "ftocmacros.h"

void kernel_revert(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
    __attribute__((annotate(RANGE_density0))) double *density0,
    __attribute__((annotate(RANGE_density1))) double *density1,
    __attribute__((annotate(RANGE_energy0))) double *energy0,
    __attribute__((annotate(RANGE_energy1))) double *energy1
) {
  int j, k;

  for (k = y_min; k <= y_max; k++) {
#pragma ivdep
    for (j = x_min; j <= x_max; j++) {
      density1[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] =
          density0[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
    }
  }

  for (k = y_min; k <= y_max; k++) {
#pragma ivdep
    for (j = x_min; j <= x_max; j++) {
      energy1[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] =
          energy0[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
    }
  }
}
