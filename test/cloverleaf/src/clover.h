// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#pragma once

extern void clover_init_comms();

extern void clover_finalize();

extern void clover_abort();

extern int clover_get_num_chunks();

extern void clover_decompose(int x_cells, int y_cells, int *left, int *right, int *bottom, int *top);

extern void clover_tile_decompose(int chunk_x_cells, int chunk_y_cells);

extern void clover_allocate_buffers();

extern void clover_deallocate_buffers();
