// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#pragma once

#include <stdbool.h>

// clang-format off
#define G_VERSION         1.3

#define G_IBIG         640000

#define G_SMALL       1.0e-16
#define G_BIG         1.0e+21

#define G_NAME_LEN_MAX    225
#define G_XDIR              1
#define G_YDIR              2

#define CHUNK_LEFT          0 // 1
#define CHUNK_RIGHT         1 // 2
#define CHUNK_BOTTOM        2 // 3
#define CHUNK_TOP           3 // 4
#define EXTERNAL_FACE      -1

#define TILE_LEFT           0 // 1
#define TILE_RIGHT          1 // 2
#define TILE_BOTTOM         2 // 3
#define TILE_TOP            3 // 4
#define EXTERNAL_TILE      -1

#define FIELD_DENSITY0      0 // 1
#define FIELD_DENSITY1      1 // 2
#define FIELD_ENERGY0       2 // 3
#define FIELD_ENERGY1       3 // 4
#define FIELD_PRESSURE      4 // 5
#define FIELD_VISCOSITY     5 // 6
#define FIELD_SOUNDSPEED    6 // 7
#define FIELD_XVEL0         7 // 8
#define FIELD_XVEL1         8 // 9
#define FIELD_YVEL0         9 // 10
#define FIELD_YVEL1        10 // 11
#define FIELD_VOL_FLUX_X   11 // 12
#define FIELD_VOL_FLUX_Y   12 // 13
#define FIELD_MASS_FLUX_X  13 // 14
#define FIELD_MASS_FLUX_Y  14 // 15
#define NUM_FIELDS         15

#define CELL_DATA           1
#define VERTEX_DATA         2
#define X_FACE_DATA         3
#define y_FACE_DATA         4

#define SOUND               1
#define X_VEL               2
#define Y_VEL               3
#define DIVERG              4

#define G_RECT              1
#define G_CIRC              2
#define G_POINT             3

#define G_LEN_MAX         500
// clang-format on

typedef struct parallel_type_t {
  bool parallel;
  bool boss;
  int max_task;
  int task;
  int boss_task;
} parallel_type;
