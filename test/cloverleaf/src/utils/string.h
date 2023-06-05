// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Niccolò Betto

#pragma once

// clang-format off
#define sswitch(string)         \
  {                             \
    const char *sel = string;   \
    do {

#define scase(value)            \
  }                             \
  if (!strcmp(sel, value)) {

#define sswitch_end             \
    } while (0);                \
  }
// clang-format on

/**
 * @brief Adjusts a string by removing leading spaces
 */
extern char *ltrim(char *s);

/**
 * @brief Adjusts a string by removing trailing spaces
 */
extern char *rtrim(char *s);

/**
 * @brief Removes leading and trailing blank characters of a string
 */
extern char *trim(char *s);

/**
 * @brief Returns the length of a character string, ignoring any trailing blanks
 */
extern int len_trim(char *s);

/**
 * @brief Returns a pointer to the end of a string (null character)
 */
extern char *strend(char *s);
