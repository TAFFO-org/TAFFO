// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#include <stdio.h>

#include "clover.h"
#include "data.h"

extern void initialise();

extern void hydro();

void clover_main() {
  clover_init_comms();

  printf("Clover Version %f\nMPI Version\nTask Count %d\n", G_VERSION, parallel.max_task);

  initialise();

  hydro();
}
