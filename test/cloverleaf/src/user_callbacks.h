// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

/*
 * Collection of functions that can be used to run custom code at specific execution points in the program.
 * By default, these functions are disabled from execution. To enable them, enable the USER_CALLBACKS toggle when
 * running make.
 */

#pragma once

#ifdef USER_CALLBACKS_ENABLED
extern void hydro_init();

extern void hydro_foreach_step(int step);

extern void hydro_done();
#else
void hydro_init() {}

void hydro_foreach_step(int step) {}

void hydro_done() {}
#endif
