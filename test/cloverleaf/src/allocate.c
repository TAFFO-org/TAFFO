// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#include <stdlib.h>

#include "definitions.h"
#include "utils/array.h"

/** @brief Allocates a 1D array, and performs shifting of the array pointer to allow access to the array using the same
 * indexing as in Fortran
 * @param array A pointer to the array pointer
 */
void allocate_array(double **array, size_t lower_bound, size_t upper_bound) {
  *array = malloc((upper_bound - lower_bound + 1) * sizeof(double));

#ifdef ARRAY_SHIFT_INDEXING
  *array = array_shift_indexing_1D_double(*array, lower_bound);
#endif
}

/** @brief Allocates a 2D array, and performs shifting of the matrix pointer to allow access to the matrix using the
 * same indexing as in Fortran
 * @param matrix A pointer to the matrix pointer
 */
void allocate_matrix(
    double **matrix, size_t lower_bound_x, size_t upper_bound_x, size_t lower_bound_y, size_t upper_bound_y
) {
  *matrix = malloc((upper_bound_y - lower_bound_y + 1) * (upper_bound_x - lower_bound_x + 1) * sizeof(double));

#ifdef ARRAY_SHIFT_INDEXING
  *matrix = array_shift_indexing_2D_double(*matrix, lower_bound_y, lower_bound_x, upper_bound_x);
#endif
}

/** @brief Dellocates a 2D array previously allocated with `allocate_array` by reverting the shifting first
 * @param array A pointer to the array pointer
 */
void deallocate_array(double **array, size_t lower_bound, size_t upper_bound) {
#ifdef ARRAY_SHIFT_INDEXING
  *array = array_revert_indexing_1D_double(*array, lower_bound);
#endif

  free(*array);
}

/** @brief Dellocates a 2D array previously allocated with `allocate_matrix` by reverting the shifting first
 * @param matrix A pointer to the matrix pointer
 */
void deallocate_matrix(
    double **matrix, size_t lower_bound_x, size_t upper_bound_x, size_t lower_bound_y, size_t upper_bound_y
) {
#ifdef ARRAY_SHIFT_INDEXING
  *matrix = array_revert_indexing_2D_double(*matrix, lower_bound_y, lower_bound_x, upper_bound_x);
#endif

  free(*matrix);
}

/**
 * @brief Allocates the data for each mesh chunk
 * @details The data fields for the mesh chunk are allocated based on the mesh size
 */
void build_field() {
  int tile, j, k;

  for (tile = 0; tile < tiles_per_chunk; tile++) {
    tile_type *cur_tile = &chunk.tiles[tile];
    field_type *cur_field = &cur_tile->field;

    allocate_matrix(
        &cur_field->density0, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    allocate_matrix(
        &cur_field->density1, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    allocate_matrix(
        &cur_field->energy0, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    allocate_matrix(
        &cur_field->energy1, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    allocate_matrix(
        &cur_field->pressure, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    allocate_matrix(
        &cur_field->viscosity, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    allocate_matrix(
        &cur_field->soundspeed, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );

    allocate_matrix(
        &cur_field->xvel0, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    allocate_matrix(
        &cur_field->xvel1, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    allocate_matrix(
        &cur_field->yvel0, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    allocate_matrix(
        &cur_field->yvel1, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );

    allocate_matrix(
        &cur_field->vol_flux_x, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    allocate_matrix(
        &cur_field->mass_flux_x, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    allocate_matrix(
        &cur_field->vol_flux_y, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    allocate_matrix(
        &cur_field->mass_flux_y, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );

    allocate_matrix(
        &cur_field->work_array1, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    allocate_matrix(
        &cur_field->work_array2, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    allocate_matrix(
        &cur_field->work_array3, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    allocate_matrix(
        &cur_field->work_array4, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    allocate_matrix(
        &cur_field->work_array5, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    allocate_matrix(
        &cur_field->work_array6, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    allocate_matrix(
        &cur_field->work_array7, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );

    allocate_array(&cur_field->cellx, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2);
    allocate_array(&cur_field->celly, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2);
    allocate_array(&cur_field->vertexx, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3);
    allocate_array(&cur_field->vertexy, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3);
    allocate_array(&cur_field->celldx, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2);
    allocate_array(&cur_field->celldy, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2);
    allocate_array(&cur_field->vertexdx, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3);
    allocate_array(&cur_field->vertexdy, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3);

    allocate_matrix(
        &cur_field->volume, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    allocate_matrix(
        &cur_field->xarea, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    allocate_matrix(
        &cur_field->yarea, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );

    // Zeroing isn't strictly neccessary but it ensures physical pages
    // are allocated. This prevents first touch overheads in the main code
    // cycle which can skew timings in the first step

    int stride = (cur_tile->t_xmax + 3) - (cur_tile->t_xmin - 2) + 1;
    for (k = 0; k <= (cur_tile->t_ymax + 3) - (cur_tile->t_ymin - 2); k++) {
      for (j = 0; j <= (cur_tile->t_xmax + 3) - (cur_tile->t_xmin - 2); j++) {
        cur_field->work_array1[k * stride + j] = 0.0;
        cur_field->work_array2[k * stride + j] = 0.0;
        cur_field->work_array3[k * stride + j] = 0.0;
        cur_field->work_array4[k * stride + j] = 0.0;
        cur_field->work_array5[k * stride + j] = 0.0;
        cur_field->work_array6[k * stride + j] = 0.0;
        cur_field->work_array7[k * stride + j] = 0.0;

        cur_field->xvel0[k * stride + j] = 0.0;
        cur_field->xvel1[k * stride + j] = 0.0;
        cur_field->yvel0[k * stride + j] = 0.0;
        cur_field->yvel1[k * stride + j] = 0.0;
      }
    }

    stride = (cur_tile->t_xmax + 2) - (cur_tile->t_xmin - 2) + 1;
    for (k = 0; k <= (cur_tile->t_ymax + 2) - (cur_tile->t_ymin - 2); k++) {
      for (j = 0; j <= (cur_tile->t_xmax + 2) - (cur_tile->t_xmin - 2); j++) {
        cur_field->density0[k * stride + j] = 0.0;
        cur_field->density1[k * stride + j] = 0.0;
        cur_field->energy0[k * stride + j] = 0.0;
        cur_field->energy1[k * stride + j] = 0.0;
        cur_field->pressure[k * stride + j] = 0.0;
        cur_field->viscosity[k * stride + j] = 0.0;
        cur_field->soundspeed[k * stride + j] = 0.0;
        cur_field->volume[k * stride + j] = 0.0;
      }
    }

    stride = (cur_tile->t_xmax + 3) - (cur_tile->t_xmin - 2) + 1;
    for (k = 0; k <= (cur_tile->t_ymax + 2) - (cur_tile->t_ymin - 2); k++) {
      for (j = 0; j <= (cur_tile->t_xmax + 3) - (cur_tile->t_xmin - 2); j++) {
        cur_field->vol_flux_x[k * stride + j] = 0.0;
        cur_field->mass_flux_x[k * stride + j] = 0.0;
        cur_field->xarea[k * stride + j] = 0.0;
      }
    }

    stride = (cur_tile->t_xmax + 2) - (cur_tile->t_xmin - 2) + 1;
    for (k = 0; k <= (cur_tile->t_ymax + 3) - (cur_tile->t_ymin - 2); k++) {
      for (j = 0; j <= (cur_tile->t_xmax + 2) - (cur_tile->t_xmin - 2); j++) {
        cur_field->vol_flux_y[k * stride + j] = 0.0;
        cur_field->mass_flux_y[k * stride + j] = 0.0;
        cur_field->yarea[k * stride + j] = 0.0;
      }
    }

    for (j = 0; j <= (cur_tile->t_xmax + 2) - (cur_tile->t_xmin - 2); j++) {
      cur_field->cellx[j] = 0.0;
      cur_field->celldx[j] = 0.0;
    }

    for (j = 0; j <= (cur_tile->t_ymax + 2) - (cur_tile->t_ymin - 2); j++) {
      cur_field->celly[j] = 0.0;
      cur_field->celldy[j] = 0.0;
    }

    for (j = 0; j <= (cur_tile->t_xmax + 3) - (cur_tile->t_xmin - 2); j++) {
      cur_field->vertexx[j] = 0.0;
      cur_field->vertexdx[j] = 0.0;
    }

    for (j = 0; j <= (cur_tile->t_ymax + 3) - (cur_tile->t_ymin - 2); j++) {
      cur_field->vertexy[j] = 0.0;
      cur_field->vertexdy[j] = 0.0;
    }
  }
}

void destroy_field() {
  for (int tile = 0; tile < tiles_per_chunk; tile++) {
    tile_type *cur_tile = &chunk.tiles[tile];
    field_type *cur_field = &cur_tile->field;

    deallocate_matrix(
        &cur_field->density0, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    deallocate_matrix(
        &cur_field->density1, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    deallocate_matrix(
        &cur_field->energy0, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    deallocate_matrix(
        &cur_field->energy1, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    deallocate_matrix(
        &cur_field->pressure, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    deallocate_matrix(
        &cur_field->viscosity, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    deallocate_matrix(
        &cur_field->soundspeed, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );

    deallocate_matrix(
        &cur_field->xvel0, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    deallocate_matrix(
        &cur_field->xvel1, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    deallocate_matrix(
        &cur_field->yvel0, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    deallocate_matrix(
        &cur_field->yvel1, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );

    deallocate_matrix(
        &cur_field->vol_flux_x, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    deallocate_matrix(
        &cur_field->mass_flux_x, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    deallocate_matrix(
        &cur_field->vol_flux_y, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    deallocate_matrix(
        &cur_field->mass_flux_y, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );

    deallocate_matrix(
        &cur_field->work_array1, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    deallocate_matrix(
        &cur_field->work_array2, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    deallocate_matrix(
        &cur_field->work_array3, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    deallocate_matrix(
        &cur_field->work_array4, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    deallocate_matrix(
        &cur_field->work_array5, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    deallocate_matrix(
        &cur_field->work_array6, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
    deallocate_matrix(
        &cur_field->work_array7, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );

    deallocate_array(&cur_field->cellx, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2);
    deallocate_array(&cur_field->celly, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2);
    deallocate_array(&cur_field->vertexx, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3);
    deallocate_array(&cur_field->vertexy, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3);
    deallocate_array(&cur_field->celldx, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2);
    deallocate_array(&cur_field->celldy, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2);
    deallocate_array(&cur_field->vertexdx, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3);
    deallocate_array(&cur_field->vertexdy, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3);

    deallocate_matrix(
        &cur_field->volume, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    deallocate_matrix(
        &cur_field->xarea, cur_tile->t_xmin - 2, cur_tile->t_xmax + 3, cur_tile->t_ymin - 2, cur_tile->t_ymax + 2
    );
    deallocate_matrix(
        &cur_field->yarea, cur_tile->t_xmin - 2, cur_tile->t_xmax + 2, cur_tile->t_ymin - 2, cur_tile->t_ymax + 3
    );
  }
}
