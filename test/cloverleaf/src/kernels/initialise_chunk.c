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
 *  @brief Driver for chunk initialisation.
 *  @author Wayne Gaudin
 *  @details Invokes the user specified chunk initialisation kernel.
 */

#include "ftocmacros.h"

void kernel_initialise_chunk(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
    double min_x,
    double min_y,
    double d_x,
    double d_y,
    double *vertexx,
    double *vertexdx,
    double *vertexy,
    double *vertexdy,
    double *cellx,
    double *celldx,
    double *celly,
    double *celldy,
    double *volume,
    double *xarea,
    double *yarea
) {
  int j, k;

#pragma ivdep
  for (j = x_min - 2; j <= x_max + 3; j++) {
    vertexx[FTNREF1D(j, x_min - 2)] = min_x + d_x * (double)(j - x_min);
  }

#pragma ivdep
  for (j = x_min - 2; j <= x_max + 3; j++) {
    vertexdx[FTNREF1D(j, x_min - 2)] = d_x;
  }

#pragma ivdep
  for (k = y_min - 2; k <= y_max + 3; k++) {
    vertexy[FTNREF1D(k, y_min - 2)] = min_y + d_y * (double)(k - y_min);
  }

#pragma ivdep
  for (k = y_min - 2; k <= y_max + 3; k++) {
    vertexdy[FTNREF1D(k, y_min - 2)] = d_y;
  }

#pragma ivdep
  for (j = x_min - 2; j <= x_max + 2; j++) {
    cellx[FTNREF1D(j, x_min - 2)] = 0.5 * (vertexx[FTNREF1D(j, x_min - 2)] + vertexx[FTNREF1D(j + 1, x_min - 2)]);
  }

#pragma ivdep
  for (j = x_min - 2; j <= x_max + 2; j++) {
    celldx[FTNREF1D(j, x_min - 2)] = d_x;
  }

#pragma ivdep
  for (k = y_min - 2; k <= y_max + 2; k++) {
    celly[FTNREF1D(k, y_min - 2)] = 0.5 * (vertexy[FTNREF1D(k, y_min - 2)] + vertexy[FTNREF1D(k + 1, x_min - 2)]);
  }

#pragma ivdep
  for (k = y_min - 2; k <= y_max + 2; k++) {
    celldy[FTNREF1D(k, y_min - 2)] = d_y;
  }

  for (k = y_min - 2; k <= y_max + 2; k++) {
#pragma ivdep
    for (j = x_min - 2; j <= x_max + 2; j++) {
      volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] = d_x * d_y;
    }
  }

  for (k = y_min - 2; k <= y_max + 2; k++) {
#pragma ivdep
    for (j = x_min - 2; j <= x_max + 2; j++) {
      xarea[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] = celldy[FTNREF1D(k, y_min - 2)];
    }
  }

  for (k = y_min - 2; k <= y_max + 2; k++) {
#pragma ivdep
    for (j = x_min - 2; j <= x_max + 2; j++) {
      yarea[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] = celldx[FTNREF1D(j, x_min - 2)];
    }
  }
}
