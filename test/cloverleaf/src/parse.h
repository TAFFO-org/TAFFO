// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#pragma once

#include <stdbool.h>
#include <stdio.h>

extern char *line;

extern int parse_init(FILE *file, const char *mask);

extern int parse_getline(const int arg);

extern char *parse_getword(const bool wrap);

extern int parse_getival(const char *word);

extern bool parse_getlval(const char *word);

extern double parse_getrval(const char *word);
