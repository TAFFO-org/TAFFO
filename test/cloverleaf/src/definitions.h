// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#pragma once

#include <stdbool.h>

#include "types/definitions.h"

extern state_type *states;  // allocatable
extern int number_of_states;

extern int step;
extern bool advect_x;

extern int tiles_per_chunk;

extern int error_condition;

extern int test_problem;
extern bool complete;

extern bool use_fortran_kernels;
extern bool use_C_kernels;
extern bool use_OA_kernels;

extern bool profiler_on;

extern profiler_type profiler;

extern double end_time;

extern int end_step;

extern double dtold;
extern double dt;
extern double time_val;
extern double dtinit;
extern double dtmin;
extern double dtmax;
extern double dtrise;
extern double dtu_safe;
extern double dtv_safe;
extern double dtc_safe;
extern double dtdiv_safe;
extern double dtc;
extern double dtu;
extern double dtv;
extern double dtdiv;

extern int visit_frequency;
extern int summary_frequency;

extern int jdt;
extern int kdt;

extern chunk_type chunk;
extern int number_of_chunks;

extern grid_type grid;
