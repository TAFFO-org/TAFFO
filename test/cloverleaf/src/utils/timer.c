// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto
// Crown Copyright (C) 2012 AWE

#include <stddef.h>
#include <sys/time.h>

double timer() {
  struct timeval t;
  gettimeofday(&t, (struct timezone *)NULL);
  return t.tv_sec + t.tv_usec * 1.0e-6;
}
