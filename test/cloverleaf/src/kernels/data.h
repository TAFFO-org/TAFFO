// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#pragma once

/**
 * @brief Constants used by the kernels. These are kept in sync with the ones in types/data.h
 * Values are offset by one as kernels use 1-based indexing
 */

// clang-format off
#define G_VERSION         1.3

#define G_IBIG         640000

#define G_SMALL       1.0e-16
#define G_BIG         1.0e+21

#define G_XDIR              1
#define G_YDIR              2

#define CHUNK_LEFT          1
#define CHUNK_RIGHT         2
#define CHUNK_BOTTOM        3
#define CHUNK_TOP           4
#define EXTERNAL_FACE      -1

#define TILE_LEFT           1
#define TILE_RIGHT          2
#define TILE_BOTTOM         3
#define TILE_TOP            4
#define EXTERNAL_TILE      -1

#define FIELD_DENSITY0      1
#define FIELD_DENSITY1      2
#define FIELD_ENERGY0       3
#define FIELD_ENERGY1       4
#define FIELD_PRESSURE      5
#define FIELD_VISCOSITY     6
#define FIELD_SOUNDSPEED    7
#define FIELD_XVEL0         8
#define FIELD_XVEL1         9
#define FIELD_YVEL0        10
#define FIELD_YVEL1        11
#define FIELD_VOL_FLUX_X   12
#define FIELD_VOL_FLUX_Y   13
#define FIELD_MASS_FLUX_X  14
#define FIELD_MASS_FLUX_Y  15
#define NUM_FIELDS         15

#define G_RECT              1
#define G_CIRC              2
#define G_POINT             3

#define G_LEN_MAX         500
// clang-format on
