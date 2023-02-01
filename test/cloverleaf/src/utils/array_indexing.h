// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Niccolò Betto

/**
 * @brief Flattens the given matrix index (2D) to the corresponding 1D index
 */
#define INDEX2D(row, column, row_size) ((row) * (row_size) + (column))
