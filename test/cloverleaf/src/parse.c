// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data.h"
#include "report.h"
#include "utils/string.h"

static FILE *file;  // iu: current file

char line_buf[100];  // Line buffer

char *line;
char *mask;  // mask: section of the file the user is interested in
char *rest;
char cur_section[100];  // here: section the parser is currently in
char active_sel[100];   // sel: active selector (*select: statement)

int parse_init(FILE *p_file, char *cmask) {
  file = p_file;
  mask = cmask;        // Set mask for which part of the file we are interested in
  line_buf[0] = '\0';  // Reset the line buffer
  line = "";
  cur_section[0] = '\0';
  active_sel[0] = '\0';

  errno = 0;
  rewind(file);
  return errno;
}

int parse_getline(const int arg) {
  while (true) {
    if (fgets(line_buf, 100, file) == NULL)  // Read in the next line
      return 1;

    // Remove invalid characters
    int len = len_trim(line_buf);
    for (int i = 0; i < len; i++)
      if (line_buf[i] < 32)
        line_buf[i] = ' ';

    line = trim(line_buf);

    // Trim the line more if a comment is there
    char *s = strchr(line, '!');
    if (s != NULL) {
      *s = '\0';
      line = trim(line);
    }
    s = strchr(line, ';');
    if (s != NULL) {
      *s = '\0';
      line = trim(line);
    }

    // Tranform to lower case
    char *c;
    if (abs(arg) != 1) {
      for (c = line; *c != '\0'; c++)
        *c = tolower(*c);
    }

    // If section-aware parsing was not requested, quit early
    if (arg < 0)
      break;

    // Make holes in the string so that it can get parsed by parse_getword
    for (char *c = line; *c != '\0'; c++)
      if (*c == '=' || *c == ',')
        *c = ' ';

    if (!strcmp(line, "*select:")) {
      report_error(
          "parse_getline", "'*select:' support is not implemented, please init the parser with a mask instead"
      );
      /*
      // Save selector and continue to the next line
      strcpy(selector, trim(line + 8));
      continue;
      */
    }

    // Check for section tags
    if (line[0] == '*') {
      char *section_tag;
      char *s = strchr(line, ' ');  // eg. "*clover leaf"
      if (s != NULL)
        *s = '\0';
      section_tag = rtrim(line);  // eg. "*clover"

      // Check if it's an end tag
      if (!strncmp(section_tag + 1, "end", 3)) {
        // Check the active section tag
        if (cur_section[0] == '\0' || strcmp(section_tag + 4, cur_section + 1))
          report_error_arg("parse_getline", "Unmatched */*end pair for section: ", cur_section + 1);
        else
          cur_section[0] = '\0';
      } else {
        // It's a new section
        strcpy(cur_section, section_tag);
      }

      continue;
    }

    // Skip if we're not in the section we're interest in
    if (strcmp(cur_section, mask))
      continue;

    break;
  }
  return 0;
}

char *parse_getword(const bool wrap) {
  // Trim leading spaces
  while (*line == ' ' && len_trim(line) > 0)
    line = trim(line + 1);

  char *word;
  // Split the line at the next space, if any
  char *s = strchr(line, ' ');
  if (s != NULL) {
    *s = '\0';
    word = trim(line);
  } else {
    word = line;
  }

  // Go to the next line if '\' is found or if wrap is set
  if (*word == '\\' || (*word == '\0' && wrap)) {
    word = "";
    if (parse_getline(0) == 0)
      word = parse_getword(wrap);
  } else {
    // Move 'line' past the word we just read, if we haven't reached the end
    if (s != NULL)
      line = s + 1;
    else
      line = strchr(line, '\0');
  }

  return word;
}

int parse_getival(const char *word) {
  int ival;
  char *end;

  errno = 0;
  ival = strtol(word, &end, 10);
  if (errno != 0)
    report_error_arg("parse_getival", "Error attempting to convert to integer: ", word);

  return ival;
}

bool parse_getlval(const char *word) {
  // clang-format off
  sswitch(word) {
    scase("on") return true;
    scase("true") return true;
    scase("off") return false;
    scase("false") return false;
  } sswitch_end;
  // clang-format on

  report_error_arg("parse_getlval", "Error attempting to convert to logical:", word);
  __builtin_unreachable();
}

double parse_getrval(const char *word) {
  double rval;
  char *end;

  errno = 0;
  rval = strtod(word, &end);
  if (errno != 0)
    report_error_arg("parse_getrval", "Error attempting to convert to real:", word);

  return rval;
}
