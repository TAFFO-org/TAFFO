// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#include <stdbool.h>
#include <stdio.h>

// Log enable
static bool log_enabled = true;

// Static test data
static FILE *file_in;
static bool fail = false;
static char fail_reason[128];

// Override report functions, clover_leaf uses these to report errors
void __attribute__((weak))
// NOLINTNEXTLINE(misc-definitions-in-headers)
report_error_arg(const char *location, const char *error, const char *arg) {
  fail = true;
  sprintf(fail_reason, "Error in %s: %s%s\n", location, error, arg);
}

void __attribute__((weak))
// NOLINTNEXTLINE(misc-definitions-in-headers)
report_error(const char *location, const char *error) {
  report_error_arg(location, error, NULL);
}

// Cleanup variables after each test
static void cleanup() {
  if (file_in != NULL)
    rewind(file_in);
  fail = false;
  fail_reason[0] = '\0';
}

#define RUN_TEST(x)         \
  do {                      \
    puts(                   \
        "\n\x1b[7m"         \
        " TEST "            \
        "\x1b[0m " #x       \
    );                      \
    x();                    \
    if (!fail) {            \
      puts(                 \
          "\x1b[32m\x1b[7m" \
          " PASS "          \
          "\x1b[0m " #x     \
      );                    \
    } else {                \
      puts(                 \
          "\x1b[31m\x1b[7m" \
          " FAIL "          \
          "\x1b[0m " #x     \
      );                    \
      puts(fail_reason);    \
    }                       \
    cleanup();              \
  } while (0)

#define LOG_PRINT(...)     \
  do {                     \
    if (log_enabled)       \
      printf(__VA_ARGS__); \
  } while (0)
