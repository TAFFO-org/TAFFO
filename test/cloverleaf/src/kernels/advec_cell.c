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
 *  @brief C cell advection kernel.
 *  @author Wayne Gaudin
 *  @details Performs a second order advective remap using van-Leer limiting
 *  with directional splitting.
 */

#include <math.h>

#include "../types/definitions.h"
#include "data.h"
#include "ftocmacros.h"

void kernel_advec_cell(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
    int dir,
    int sweep_number,
    __attribute__((annotate(RANGE_vertexdx))) double *vertexdx,
    __attribute__((annotate(RANGE_vertexdy))) double *vertexdy,
    __attribute__((annotate(RANGE_volume))) double *volume,
    __attribute__((annotate(RANGE_density1))) double *density1,
    __attribute__((annotate(RANGE_energy1))) double *energy1,
    __attribute__((annotate(RANGE_mass_flux_x))) double *mass_flux_x,
    __attribute__((annotate(RANGE_vol_flux_x))) double *vol_flux_x,
    __attribute__((annotate(RANGE_mass_flux_y))) double *mass_flux_y,
    __attribute__((annotate(RANGE_vol_flux_y))) double *vol_flux_y,
    __attribute__((annotate(RANGE_work_array1))) double *pre_vol,
    __attribute__((annotate(RANGE_work_array2))) double *post_vol,
    __attribute__((annotate(RANGE_work_array3))) double *pre_mass,
    __attribute__((annotate(RANGE_work_array4))) double *post_mass,
    __attribute__((annotate(RANGE_work_array5))) double *advec_vol,
    __attribute__((annotate(RANGE_work_array6))) double *post_ener,
    __attribute__((annotate(RANGE_work_array7))) double *ener_flux
) {
  int j, k, upwind, donor, downwind, dif;

  __attribute__((annotate("scalar()"))) double sigma, sigmat, sigmav, sigmam, sigma3, sigma4, diffuw, diffdw, limiter;
  __attribute__((annotate("scalar()"))) double one_by_six;

  one_by_six = 1.0 / 6.0;

  if (dir == G_XDIR) {
    if (sweep_number == 1) {
      for (k = y_min - 2; k <= y_max + 2; k++) {
#pragma ivdep
        for (j = x_min - 2; j <= x_max + 2; j++) {
          pre_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
              volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] +
              (vol_flux_x[FTNREF2D(j + 1, k, x_max + 5, x_min - 2, y_min - 2)] -
               vol_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] +
               vol_flux_y[FTNREF2D(j, k + 1, x_max + 4, x_min - 2, y_min - 2)] -
               vol_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)]);
          post_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
              pre_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] -
              (vol_flux_x[FTNREF2D(j + 1, k, x_max + 5, x_min - 2, y_min - 2)] -
               vol_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)]);
        }
      }

    } else {
      for (k = y_min - 2; k <= y_max + 2; k++) {
#pragma ivdep
        for (j = x_min - 2; j <= x_max + 2; j++) {
          pre_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
              volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] +
              vol_flux_x[FTNREF2D(j + 1, k, x_max + 5, x_min - 2, y_min - 2)] -
              vol_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)];
          post_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
              volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    for (k = y_min; k <= y_max; k++) {
      for (j = x_min; j <= x_max + 2; j++) {
        if (vol_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] > 0.0) {
          upwind = j - 2;
          donor = j - 1;
          downwind = j;
          dif = donor;
        } else {
          upwind = MIN(j + 1, x_max + 2);
          donor = j;
          downwind = j - 1;
          dif = upwind;
        }

        sigmat = fabs(
            vol_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] /
            pre_vol[FTNREF2D(donor, k, x_max + 5, x_min - 2, y_min - 2)]
        );
        sigma3 = (1.0 + sigmat) * (vertexdx[FTNREF1D(j, x_min - 2)] / vertexdx[FTNREF1D(dif, x_min - 2)]);
        sigma4 = 2.0 - sigmat;

        sigma = sigmat;
        sigmav = sigmat;

        diffuw = density1[FTNREF2D(donor, k, x_max + 4, x_min - 2, y_min - 2)] -
                 density1[FTNREF2D(upwind, k, x_max + 4, x_min - 2, y_min - 2)];
        diffdw = density1[FTNREF2D(downwind, k, x_max + 4, x_min - 2, y_min - 2)] -
                 density1[FTNREF2D(donor, k, x_max + 4, x_min - 2, y_min - 2)];
        if (diffuw * diffdw > 0.0) {
          limiter = (1.0 - sigmav) * SIGN(1.0, diffdw) *
                    MIN(fabs(diffuw), MIN(fabs(diffdw), one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
        } else {
          limiter = 0.0;
        }
        mass_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
            vol_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] *
            (density1[FTNREF2D(donor, k, x_max + 4, x_min - 2, y_min - 2)] + limiter);

        sigmam = fabs(mass_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)]) /
                 (density1[FTNREF2D(donor, k, x_max + 4, x_min - 2, y_min - 2)] *
                  pre_vol[FTNREF2D(donor, k, x_max + 5, x_min - 2, y_min - 2)]);
        diffuw = energy1[FTNREF2D(donor, k, x_max + 4, x_min - 2, y_min - 2)] -
                 energy1[FTNREF2D(upwind, k, x_max + 4, x_min - 2, y_min - 2)];
        diffdw = energy1[FTNREF2D(downwind, k, x_max + 4, x_min - 2, y_min - 2)] -
                 energy1[FTNREF2D(donor, k, x_max + 4, x_min - 2, y_min - 2)];
        if (diffuw * diffdw > 0.0) {
          limiter = (1.0 - sigmam) * SIGN(1.0, diffdw) *
                    MIN(fabs(diffuw), MIN(fabs(diffdw), one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
        } else {
          limiter = 0.0;
        }
        ener_flux[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
            mass_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] *
            (energy1[FTNREF2D(donor, k, x_max + 4, x_min - 2, y_min - 2)] + limiter);
      }
    }

    for (k = y_min; k <= y_max; k++) {
#pragma ivdep
      for (j = x_min; j <= x_max; j++) {
        pre_mass[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
            density1[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] *
            pre_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)];
        post_mass[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
            pre_mass[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] +
            mass_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] -
            mass_flux_x[FTNREF2D(j + 1, k, x_max + 5, x_min - 2, y_min - 2)];
        post_ener[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
            (energy1[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] *
                 pre_mass[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] +
             ener_flux[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] -
             ener_flux[FTNREF2D(j + 1, k, x_max + 5, x_min - 2, y_min - 2)]) /
            post_mass[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)];
        advec_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
            pre_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] +
            vol_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] -
            vol_flux_x[FTNREF2D(j + 1, k, x_max + 5, x_min - 2, y_min - 2)];

        density1[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] =
            post_mass[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] /
            advec_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)];
        energy1[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] =
            post_ener[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }

  } else if (dir == G_YDIR) {
    if (sweep_number == 1) {
      for (k = y_min - 2; k <= y_max + 2; k++) {
#pragma ivdep
        for (j = x_min - 2; j <= x_max + 2; j++) {
          pre_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
              volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] +
              (vol_flux_y[FTNREF2D(j, k + 1, x_max + 4, x_min - 2, y_min - 2)] -
               vol_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] +
               vol_flux_x[FTNREF2D(j + 1, k, x_max + 5, x_min - 2, y_min - 2)] -
               vol_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)]);
          post_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
              pre_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] -
              (vol_flux_y[FTNREF2D(j, k + 1, x_max + 4, x_min - 2, y_min - 2)] -
               vol_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)]);
        }
      }

    } else {
      for (k = y_min - 2; k <= y_max + 2; k++) {
#pragma ivdep
        for (j = x_min - 2; j <= x_max + 2; j++) {
          pre_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
              volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] +
              vol_flux_y[FTNREF2D(j, k + 1, x_max + 4, x_min - 2, y_min - 2)] -
              vol_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
          post_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
              volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
        }
      }
    }

    for (k = y_min; k <= y_max + 2; k++) {
      for (j = x_min; j <= x_max; j++) {
        if (vol_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] > 0.0) {
          upwind = k - 2;
          donor = k - 1;
          downwind = k;
          dif = donor;
        } else {
          upwind = MIN(k + 1, y_max + 2);
          donor = k;
          downwind = k - 1;
          dif = upwind;
        }

        sigmat = fabs(
            vol_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] /
            pre_vol[FTNREF2D(j, donor, x_max + 5, x_min - 2, y_min - 2)]
        );
        sigma3 = (1.0 + sigmat) * (vertexdy[FTNREF1D(k, y_min - 2)] / vertexdy[FTNREF1D(dif, y_min - 2)]);
        sigma4 = 2.0 - sigmat;

        sigma = sigmat;
        sigmav = sigmat;

        diffuw = density1[FTNREF2D(j, donor, x_max + 4, x_min - 2, y_min - 2)] -
                 density1[FTNREF2D(j, upwind, x_max + 4, x_min - 2, y_min - 2)];
        diffdw = density1[FTNREF2D(j, downwind, x_max + 4, x_min - 2, y_min - 2)] -
                 density1[FTNREF2D(j, donor, x_max + 4, x_min - 2, y_min - 2)];

        if (diffuw * diffdw > 0.0) {
          limiter = (1.0 - sigmav) * SIGN(1.0, diffdw) *
                    MIN(fabs(diffuw), MIN(fabs(diffdw), one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
        } else {
          limiter = 0.0;
        }
        mass_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] =
            vol_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] *
            (density1[FTNREF2D(j, donor, x_max + 4, x_min - 2, y_min - 2)] + limiter);

        sigmam = fabs(mass_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)]) /
                 (density1[FTNREF2D(j, donor, x_max + 4, x_min - 2, y_min - 2)] *
                  pre_vol[FTNREF2D(j, donor, x_max + 5, x_min - 2, y_min - 2)]);
        diffuw = energy1[FTNREF2D(j, donor, x_max + 4, x_min - 2, y_min - 2)] -
                 energy1[FTNREF2D(j, upwind, x_max + 4, x_min - 2, y_min - 2)];
        diffdw = energy1[FTNREF2D(j, downwind, x_max + 4, x_min - 2, y_min - 2)] -
                 energy1[FTNREF2D(j, donor, x_max + 4, x_min - 2, y_min - 2)];
        if (diffuw * diffdw > 0.0) {
          limiter = (1.0 - sigmam) * SIGN(1.0, diffdw) *
                    MIN(fabs(diffuw), MIN(fabs(diffdw), one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
        } else {
          limiter = 0.0;
        }
        ener_flux[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
            mass_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] *
            (energy1[FTNREF2D(j, donor, x_max + 4, x_min - 2, y_min - 2)] + limiter);
      }
    }

    for (k = y_min; k <= y_max; k++) {
#pragma ivdep
      for (j = x_min; j <= x_max; j++) {
        pre_mass[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
            density1[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] *
            pre_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)];
        post_mass[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
            pre_mass[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] +
            mass_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] -
            mass_flux_y[FTNREF2D(j, k + 1, x_max + 4, x_min - 2, y_min - 2)];
        post_ener[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
            (energy1[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] *
                 pre_mass[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] +
             ener_flux[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] -
             ener_flux[FTNREF2D(j, k + 1, x_max + 5, x_min - 2, y_min - 2)]) /
            post_mass[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)];
        advec_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
            pre_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] +
            vol_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] -
            vol_flux_y[FTNREF2D(j, k + 1, x_max + 4, x_min - 2, y_min - 2)];

        density1[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] =
            post_mass[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] /
            advec_vol[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)];
        energy1[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] =
            post_ener[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)];
      }
    }
  }
}
