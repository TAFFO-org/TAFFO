// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Niccolò Betto

#pragma once

#define DENSITY0 0
#define DENSITY1 1
#define ENERGY0 2
#define ENERGY1 3
#define PRESSURE 4
#define VISCOSITY 5
#define SOUNDSPEED 6
#define XVEL0 7
#define XVEL1 8
#define YVEL0 9
#define YVEL1 10
#define VOL_FLUX_X 11
#define MASS_FLUX_X 12
#define VOL_FLUX_Y 13
#define MASS_FLUX_Y 14
#define WORK_ARRAY1 15
#define WORK_ARRAY2 16
#define WORK_ARRAY3 17
#define WORK_ARRAY4 18
#define WORK_ARRAY5 19
#define WORK_ARRAY6 20
#define WORK_ARRAY7 21
#define CELLX 22
#define CELLY 23
#define VERTEXX 24
#define VERTEXY 25
#define CELLDX 26
#define CELLDY 27
#define VERTEXDX 28
#define VERTEXDY 29
#define VOLUME 30
#define XAREA 31
#define YAREA 32
#define USAGE_INFO_SIZE 33

typedef struct usage_info_t {
  double min;
  double max;
  char *array_name;
} usage_info;

extern void init_usage_tracker();

extern void close_usage_tracker();

extern void print_annotations();

extern void print_usage_info();

extern void sample_usage_info();
