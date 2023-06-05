// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Niccolò Betto

/*
 * Utility functions to dump the program's arrays to file
 */

#pragma once

#include <errno.h>
#include <stdio.h>

#include "../definitions.h"

#define RANGE_DEF(array) extern void range_##array(usage_info *info);

#define RANGE_2D_DEF(array, xmax, xmin, ymax, ymin) RANGE_DEF(array);

#define RANGE_1DX_DEF(array, xmax, xmin) RANGE_DEF(array);

#define RANGE_1DY_DEF(array, ymax, ymin) RANGE_DEF(array);

#define ARRAY_2D_DEF(array, xmax, xmin, ymax, ymin) RANGE_2D_DEF(array, xmax, xmin, ymax, ymin)

#define ARRAY_1DX_DEF(array, xmax, xmin) RANGE_1DX_DEF(array, xmax, xmin)

#define ARRAY_1DY_DEF(array, ymax, ymin) RANGE_1DY_DEF(array, ymax, ymin)

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

#undef RANGE_DEF
#undef RANGE_2D_DEF
#undef RANGE_1DX_DEF
#undef RANGE_1DY_DEF
#undef ARRAY_2D_DEF
#undef ARRAY_1DX_DEF
#undef ARRAY_1DY_DEF
