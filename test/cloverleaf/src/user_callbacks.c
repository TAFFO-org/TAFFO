// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#include "utils/usage_tracker.h"

#ifdef USER_CALLBACKS_ENABLED

void hydro_init() {
  init_usage_tracker();
}

void hydro_foreach_step(int step) {
  sample_usage_info();
}

void hydro_done() {
  // Sample one more time to get the final values
  sample_usage_info();

  print_usage_info();
  print_annotations();

  close_usage_tracker();
}

#endif
