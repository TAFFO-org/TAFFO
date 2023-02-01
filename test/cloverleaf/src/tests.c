// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#include "tests.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "data.h"
#include "definitions.h"
#include "parse.h"
#include "utils/array.h"

void test_parse_getword() {
  char test[16] = " test_problem 2\0";

  parse_init(file_in, "");
  line = test;
  LOG_PRINT("Parsing: '%s'\n", line);

  fflush(stdout);

  char *word = parse_getword(false);
  LOG_PRINT("Result: '%s'\n", word);
}

/**
 * @brief Print the whole clover.in file using parse_getline
 */
void test_parse_getline() {
  parse_init(file_in, "*clover");

  while (parse_getline(0) == 0) {
    LOG_PRINT("Parsing: '%s'\n", line);
  }
}

void test_parse_getline_getword() {
  parse_init(file_in, "*clover");

  while (parse_getline(0) == 0) {
    char l[100];
    strcpy(l, line);
    char *word = parse_getword(false);
    LOG_PRINT("Word: '%s' in line '%s'\n", word, l);
  }
  parse_getword(true);
}

extern void read_input();

void test_parse_file() {
  parallel.boss = true;
  parallel.max_task = 1;

  LOG_PRINT("Reading clover.in\n");
  g_in = file_in;
  g_out = stdout;
  read_input();
  // `states` is allocated inside read_input() so we need to free it
  free(states);
}

void test_timestep_print() {
  LOG_PRINT(
      "Step %7d time %11.7lf control %11s timestep  %9.2e%8d, %8d x %9.2e y %9.2e\n",
      1,
      0.0000000,
      "sound",
      6.16e-03,
      1,
      1,
      6.01e-154,
      6.01e-154
  );
  LOG_PRINT(
      "Step %7d time %11.7lf control %11s timestep  %9.2e%8d, %8d x %9.2e y %9.2e\n",
      2,
      0.0061626,
      "sound",
      3.76e-03,
      1,
      1,
      3.81e-320,
      6.94e-310
  );
  LOG_PRINT(
      "Step %7d time %11.7lf control %11s timestep  %9.2e%8d, %8d x %9.2e y %9.2e\n",
      3,
      0.0099242,
      "sound",
      4.78e-03,
      1,
      1,
      3.81e-320,
      4.76e-321
  );
}

void test_relative_array_indexing_1D() {
  int i = 0, j = 0;
  int t_xmax = 3;
  int t_xmin = 1;

  size_t size = (t_xmax + 2) - (t_xmin - 2) + 1; // upper_bound - lower_bound + 1
  double *array = calloc(size, sizeof(double));

  // Set boundary values
  array[0] = 1.f;
  array[size - 1] = 2.f;

  LOG_PRINT("Array contents:\n");

  // Print array contents
  for (i = 0; i < size; i++)
    LOG_PRINT("[%2d]%4.1f\n", i, array[i]);
  LOG_PRINT("\n");

  LOG_PRINT("Array contents by index fiddling:\n");

  // Print array by index fiddling
  for (i = t_xmin - 2; i <= t_xmax + 2; i++)
    LOG_PRINT("[%2d]%4.1f\n", i - (t_xmin - 2), array[i - (t_xmin - 2)]);
  LOG_PRINT("\n");

  LOG_PRINT("Array contents by pointer modification:\n");

  // Print array contents by modifying pointer
  double *array2 = array_shift_indexing_1D_double(array, t_xmin - 2);

  for (i = t_xmin - 2; i <= t_xmax + 2; i++)
    LOG_PRINT("[%2d]%4.1f\n", i, array2[i]);

  free(array);
}

void test_relative_array_indexing_2D() {
  int x = 0, y = 0;

  // These values yield a 6x8 matrix
  int t_xmax = 4;
  int t_xmin = 1;
  int t_ymax = 2;
  int t_ymin = 1;

  // Create a reference matrix to compare with
  double reference[6][8] = {0};
  reference[0][0] = 1.f;
  reference[0][7] = 2.f;
  reference[1][7] = 2.f;
  reference[2][7] = 2.f;
  reference[3][7] = 2.f;
  reference[4][7] = 2.f;
  reference[5][7] = 3.f;

  LOG_PRINT("Reference matrix: [index] value\n");

  for (y = 0; y < 6; y++) {
    for (x = 0; x < 8; x++)
      LOG_PRINT("[%2d]%4.1f  ", y * 8 + x, reference[y][x]);
    LOG_PRINT("\n");
  }
  LOG_PRINT("\n");

  // Elements in a column
  int col_size = ((t_ymax + 2) - (t_ymin - 2) + 1); // upper_bound_y - lower_bound_y + 1
  // Elements in a row
  int row_size = ((t_xmax + 2) - (t_xmin - 2) + 1); // upper_bound_x - lower_bound_x + 1

  double *matrix = calloc(col_size * row_size, sizeof(double));

  LOG_PRINT("Matrix size: %d x %d = %d\n", col_size, row_size, col_size * row_size);

  // Fill the matrix with row endings
  matrix[0] = 1.f;
  matrix[0 * row_size + 7] = 2.f;
  matrix[1 * row_size + 7] = 2.f;
  matrix[2 * row_size + 7] = 2.f;
  matrix[3 * row_size + 7] = 2.f;
  matrix[4 * row_size + 7] = 2.f;
  matrix[5 * row_size + 7] = 3.f;

  LOG_PRINT("Matrix contents: [index] value\n");

  // Print matrix contents
  for (y = 0; y < col_size; y++) { // for every row
    for (x = 0; x < row_size; x++) // for every element in the row
      LOG_PRINT("[%2d]%4.1f  ", y * row_size + x, matrix[y * row_size + x]);
    LOG_PRINT("\n");
  }
  LOG_PRINT("\n");

  LOG_PRINT("Matrix contents by index fiddling: [index] value\n");

  // Print array by index fiddling
  for (y = t_ymin - 2; y <= t_ymax + 2; y++) { // for every row
    for (x = t_xmin - 2; x <= t_xmax + 2; x++) // for every element in the row
      LOG_PRINT(
          "[%2d]%4.1f  ",
          (y - (t_ymin - 2)) * row_size + (x - (t_xmin - 2)),
          matrix[(y - (t_ymin - 2)) * row_size + (x - (t_xmin - 2))]
      );
    LOG_PRINT("\n");
  }
  LOG_PRINT("\n");

  LOG_PRINT("Matrix contents by pointer modification: [index] value\n");

  // Print array contents by modifying pointer
  // ptr - lower_bound_y * row_size - lower_bound_x
  double *matrix2 = array_shift_indexing_2D_double(matrix, t_ymin - 2, t_xmin - 2, t_xmax + 2);

  for (y = t_ymin - 2; y <= t_ymax + 2; y++) { // for every row
    for (x = t_xmin - 2; x <= t_xmax + 2; x++) // for every element in the row
      LOG_PRINT("[%2d]%4.1f  ", y * row_size + x, matrix2[y * row_size + x]);
    LOG_PRINT("\n");
  }

  free(matrix);
}

extern void build_field();
extern void destroy_field();

void test_build_field() {
  tiles_per_chunk = 1;
  LOG_PRINT(
      "Allocating %d tiles per chunk, total size: %zu bytes\n", tiles_per_chunk, tiles_per_chunk * sizeof(tile_type)
  );
  chunk.tiles = malloc(tiles_per_chunk * sizeof(tile_type));

  LOG_PRINT("Building field...\n");
  build_field();
  LOG_PRINT("Done allocating\n");

  LOG_PRINT("Destroying fields...\n");
  destroy_field();
  free(chunk.tiles);
  LOG_PRINT("Done deallocating\n");
}

void test_build_field_stress() {
  // Should take about 20 second to run on a reasonably fast machine
  // Ensures that the memory is freed properly
  LOG_PRINT("Running build_field test 10 million times, hold tight...\n");

  // Save previous log state
  bool prev_log_enabled = log_enabled;
  // Silence log to avoid spamming the console (also makes the test run much faster)
  log_enabled = false;

  clock_t start = clock();
  for (int i = 0; i < 10000000; i++) {
    test_build_field();
  }

  // Restore log
  log_enabled = prev_log_enabled;
  LOG_PRINT("Done.\nTook %.3f seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC);
}

int main(int argc, char **argv) {
  puts("*** CLoverLeaf unit test runner ***");
  file_in = fopen("clover.in", "r");

  RUN_TEST(test_parse_getword);
  RUN_TEST(test_parse_getline);
  RUN_TEST(test_parse_getline_getword);
  RUN_TEST(test_parse_file);
  RUN_TEST(test_timestep_print);
  RUN_TEST(test_relative_array_indexing_1D);
  RUN_TEST(test_relative_array_indexing_2D);
  RUN_TEST(test_build_field);
  RUN_TEST(test_build_field_stress);

  puts("\nAll tests passed!");
  return 0;
}
