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

double dtold;
double dt;
double time_val;
double dtinit;
double dtmin;
double dtmax;
double dtrise;
double dtu_safe;
double dtv_safe;
double dtc_safe;
double dtdiv_safe;
double dtc;
double dtu;
double dtv;
double dtdiv;

int visit_frequency;
int summary_frequency;

int jdt;
int kdt;

chunk_type chunk;
int number_of_chunks;

grid_type grid;
