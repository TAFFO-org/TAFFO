// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Niccolò Betto

#include <ctype.h>
#include <string.h>

char *ltrim(char *s) {
  while (isspace(*s))
    s++;
  return s;
}

char *rtrim(char *s) {
  char *back = s + strlen(s);
  while (isspace(*--back))
    ;
  *(back + 1) = '\0';
  return s;
}

char *trim(char *s) { return rtrim(ltrim(s)); }

int len_trim(char *s) {
  int len = strlen(s);
  char *back = s + len;
  while (isspace(*--back))
    len--;
  return len;
}

char *strend(char *s) { return s + strlen(s); }
