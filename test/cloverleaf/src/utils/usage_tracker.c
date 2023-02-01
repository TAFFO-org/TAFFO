// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Niccolò Betto

/*
 * UsageTracker: periodically scan the program's arrays for min and max values
 *
 * The usage tracker periodically reads the program's arrays looking for the max and min values used.
 * It can then print a report of the collected data. Additionally, it has support for directly printing TAFFO annotation.
 */

#include "usage_tracker.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../definitions.h"
#include "range.h"

#define USAGE_TRACKER_FILE "usage_tracker.txt"

static FILE *usage_tracker_file = NULL;
static usage_info usage_data[USAGE_INFO_SIZE];

static int sample_count;

void init_usage_tracker() {
  if (usage_tracker_file != NULL) {
    fclose(usage_tracker_file);
    usage_tracker_file = NULL;
  }

  sample_count = 0;
  memset(usage_data, 0, sizeof(usage_data));

  errno = 0;
  usage_tracker_file = fopen(USAGE_TRACKER_FILE, "w");
  if (errno != 0) {
    printf("Error opening " USAGE_TRACKER_FILE " file, output will go to stdout.\n");
    usage_tracker_file = stdout;
  }
}

void close_usage_tracker() {
  if (usage_tracker_file != NULL && usage_tracker_file != stdout) {
    fclose(usage_tracker_file);
  }
}

void print_usage_info() {
  const char *report_format = "%s:\n * Min: %.8f\n * Max: %.8f\n";

  fprintf(usage_tracker_file, "Array usage report\n------------------\n");
  fprintf(usage_tracker_file, "Stats:\n - Sample count: %d\n\n", sample_count);

  for (int i = 0; i < USAGE_INFO_SIZE; i++) {
    usage_info *info = &usage_data[i];
    fprintf(usage_tracker_file, report_format, info->array_name, info->min, info->max);
  }
  fprintf(usage_tracker_file, "\n");
}

void print_annotations() {
  const char *annotation_format = "%-12s | __attribute__((annotate(\"scalar(range(%4d,%4d) final)\")))\n";

  fprintf(usage_tracker_file, "Ready-to-use annotations\n------------------------\n");
  for (int i = 0; i < USAGE_INFO_SIZE; i++) {
    usage_info *info = &usage_data[i];
    fprintf(usage_tracker_file, annotation_format, info->array_name, (int)floor(info->min), (int)ceil(info->max));
  }
  fprintf(usage_tracker_file, "\n");
}

void sample_usage_info() {
  range_density0(&usage_data[DENSITY0]);
  range_density1(&usage_data[DENSITY1]);
  range_energy0(&usage_data[ENERGY0]);
  range_energy1(&usage_data[ENERGY1]);
  range_pressure(&usage_data[PRESSURE]);
  range_viscosity(&usage_data[VISCOSITY]);
  range_soundspeed(&usage_data[SOUNDSPEED]);
  range_xvel0(&usage_data[XVEL0]);
  range_xvel1(&usage_data[XVEL1]);
  range_yvel0(&usage_data[YVEL0]);
  range_yvel1(&usage_data[YVEL1]);
  range_vol_flux_x(&usage_data[VOL_FLUX_X]);
  range_mass_flux_x(&usage_data[MASS_FLUX_X]);
  range_vol_flux_y(&usage_data[VOL_FLUX_Y]);
  range_mass_flux_y(&usage_data[MASS_FLUX_Y]);
  range_work_array1(&usage_data[WORK_ARRAY1]);
  range_work_array2(&usage_data[WORK_ARRAY2]);
  range_work_array3(&usage_data[WORK_ARRAY3]);
  range_work_array4(&usage_data[WORK_ARRAY4]);
  range_work_array5(&usage_data[WORK_ARRAY5]);
  range_work_array6(&usage_data[WORK_ARRAY6]);
  range_work_array7(&usage_data[WORK_ARRAY7]);
  range_cellx(&usage_data[CELLX]);
  range_celly(&usage_data[CELLY]);
  range_vertexx(&usage_data[VERTEXX]);
  range_vertexy(&usage_data[VERTEXY]);
  range_celldx(&usage_data[CELLDX]);
  range_celldy(&usage_data[CELLDY]);
  range_vertexdx(&usage_data[VERTEXDX]);
  range_vertexdy(&usage_data[VERTEXDY]);
  range_volume(&usage_data[VOLUME]);
  range_xarea(&usage_data[XAREA]);
  range_yarea(&usage_data[YAREA]);
  sample_count++;
}
