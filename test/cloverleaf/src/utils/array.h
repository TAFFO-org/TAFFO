// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Niccolò Betto

#pragma once

#include <stddef.h>

/**
 * @brief Utilities for converting C array pointers to Fortran arrays with arbitrary lower/upper bounds. The size of the
 * C array must match the one of the Fortran array, to not incur in OOB reads.
 *
 * For the purpose of representing a matrix of points in the x-y plane, arrays should be viewed as follows:
 * A[y][x]
 * this guarantees that the quicker changing index is the x one, and the slower changing one is the y one.
 */

/**
 * @brief Shifts the given array indexing range to [lower_bound, N - lower_bound]
 * @param ptr A pointer to the array, with indexing range [0, N]
 * @returns A pointer to the original array, with indexing range [lower_bound, N - lower_bound]
 */
extern int *array_shift_indexing_1D_int(int *ptr, size_t lower_bound);

/**
 * @brief Shifts the given array indexing range to [lower_bound, N - lower_bound]
 * @param ptr A pointer to the array, with indexing range [0, N]
 * @returns A pointer to the original array, with indexing range [lower_bound, N - lower_bound]
 */
extern float *array_shift_indexing_1D_float(float *ptr, size_t lower_bound);

/**
 * @brief Shifts the given array indexing range to [lower_bound, N - lower_bound]
 * @param ptr A pointer to the array, with indexing range [0, N]
 * @returns A pointer to the original array, with indexing range [lower_bound, N - lower_bound]
 */
extern double *array_shift_indexing_1D_double(double *ptr, size_t lower_bound);

/**
 * @brief Reverts the shift performed by `array_shift_indexing_1D_*` on the given array
 * @param ptr A pointer to the array, with indexing range [lower_bound, N - lower_bound]
 * @returns A pointer to the original array, with indexing range [0, N]
 */
extern int *array_revert_indexing_1D_int(int *ptr, size_t lower_bound);

/**
 * @brief Reverts the shift performed by `array_shift_indexing_1D_*` on the given array
 * @param ptr A pointer to the array, with indexing range [lower_bound, N - lower_bound]
 * @returns A pointer to the original array, with indexing range [0, N]
 */
extern float *array_revert_indexing_1D_float(float *ptr, size_t lower_bound);

/**
 * @brief Reverts the shift performed by `array_shift_indexing_1D_*` on the given array
 * @param ptr A pointer to the array, with indexing range [lower_bound, N - lower_bound]
 * @returns A pointer to the original array, with indexing range [0, N]
 */
extern double *array_revert_indexing_1D_double(double *ptr, size_t lower_bound);

/**
 * @brief Shifts the given 2-dimensional array indexing range to [lower_bound_x, N - lower_bound_x] for the first
 * dimension and [lower_bound_y, N - lower_bound_y] for the second dimension
 * @param ptr A pointer to the array, with indexing range [0, N], [0, N]
 * @returns A pointer to the original 2-dimensional array, with indexing range [lower_bound_x, N - lower_bound_x] for
 * the first dimension and [lower_bound_y, N - lower_bound_y] for the second dimension
 */
extern int *array_shift_indexing_2D_int(int *ptr, size_t lower_bound_y, size_t lower_bound_x, size_t upper_bound_x);

/**
 * @brief Shifts the given 2-dimensional array indexing range to [lower_bound_x, N - lower_bound_x] for the first
 * dimension and [lower_bound_y, N - lower_bound_y] for the second dimension
 * @param ptr A pointer to the array, with indexing range [0, N], [0, N]
 * @returns A pointer to the original 2-dimensional array, with indexing range [lower_bound_x, N - lower_bound_x] for
 * the first dimension and [lower_bound_y, N - lower_bound_y] for the second dimension
 */
extern float *array_shift_indexing_2D_float(float *ptr, size_t lower_bound_y, size_t lower_bound_x,
                                            size_t upper_bound_x);

/**
 * @brief Shifts the given 2-dimensional array indexing range to [lower_bound_x, N - lower_bound_x] for the first
 * dimension and [lower_bound_y, N - lower_bound_y] for the second dimension
 * @param ptr A pointer to the array, with indexing range [0, N], [0, N]
 * @returns A pointer to the original 2-dimensional array, with indexing range [lower_bound_x, N - lower_bound_x] for
 * the first dimension and [lower_bound_y, N - lower_bound_y] for the second dimension
 */
extern double *array_shift_indexing_2D_double(double *ptr, size_t lower_bound_y, size_t lower_bound_x,
                                              size_t upper_bound_x);

/**
 * @brief Reverts the shift performed by `array_shift_indexing_2D_*` on the given 2-dimensional array
 * @param ptr A pointer to the array, with indexing range [lower_bound_x, N - lower_bound_x] for the first dimension and
 * [lower_bound_y, N - lower_bound_y] for the second dimension
 * @returns A pointer to the original 2-dimensional array, with indexing range [0, N], [0, N]
 */
extern int *array_revert_indexing_2D_int(int *ptr, size_t lower_bound_y, size_t lower_bound_x, size_t upper_bound_x);

/**
 * @brief Reverts the shift performed by `array_shift_indexing_2D_*` on the given 2-dimensional array
 * @param ptr A pointer to the array, with indexing range [lower_bound_x, N - lower_bound_x] for the first dimension and
 * [lower_bound_y, N - lower_bound_y] for the second dimension
 * @returns A pointer to the original 2-dimensional array, with indexing range [0, N], [0, N]
 */
extern float *array_revert_indexing_2D_float(float *ptr, size_t lower_bound_y, size_t lower_bound_x,
                                             size_t upper_bound_x);

/**
 * @brief Reverts the shift performed by `array_shift_indexing_2D_*` on the given 2-dimensional array
 * @param ptr A pointer to the array, with indexing range [lower_bound_x, N - lower_bound_x] for the first dimension and
 * [lower_bound_y, N - lower_bound_y] for the second dimension
 * @returns A pointer to the original 2-dimensional array, with indexing range [0, N], [0, N]
 */
extern double *array_revert_indexing_2D_double(double *ptr, size_t lower_bound_y, size_t lower_bound_x,
                                               size_t upper_bound_x);
