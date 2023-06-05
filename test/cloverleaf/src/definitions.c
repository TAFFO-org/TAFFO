// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#include "types/definitions.h"

state_type *states;
int number_of_states;

int step;
bool advect_x;

int tiles_per_chunk;

int error_condition;

int test_problem;
bool complete;

bool use_fortran_kernels;
bool use_C_kernels;
bool use_OA_kernels;

bool profiler_on;

profiler_type profiler;

double end_time;

int end_step;

double dt;  // This variable is used in contexts where values are too small to be represented correctly
            // with fixed point notation, therefore it is not annotated
__attribute__((annotate("scalar(range(0,1))"))) double dtold;
__attribute__((annotate("scalar(range(0,1))"))) double time_val;
__attribute__((annotate("scalar(range(0,1))"))) double dtinit;
__attribute__((annotate("scalar(range(0,1))"))) double dtmin;
__attribute__((annotate("scalar(range(0,1))"))) double dtmax;
__attribute__((annotate("scalar(range(0,2))"))) double dtrise;
__attribute__((annotate("scalar(range(0,1))"))) double dtu_safe;
__attribute__((annotate("scalar(range(0,1))"))) double dtv_safe;
__attribute__((annotate("scalar(range(0,1))"))) double dtc_safe;
__attribute__((annotate("scalar(range(0,1))"))) double dtdiv_safe;
__attribute__((annotate("scalar(range(0,1))"))) double dtc;
__attribute__((annotate("scalar(range(0,1))"))) double dtu;
__attribute__((annotate("scalar(range(0,1))"))) double dtv;
__attribute__((annotate("scalar(range(0,1))"))) double dtdiv;

int visit_frequency;
int summary_frequency;

int jdt;
int kdt;

chunk_type __attribute__((annotate(ANNOTATION_CHUNK_TYPE))) chunk;
int number_of_chunks;

grid_type grid;
