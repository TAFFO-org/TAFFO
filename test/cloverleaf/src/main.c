// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2022 Niccolò Betto

extern void clover_main();

/*
 * This is the entry point for the program
 * It is defined as a weak symbol so that it can be overridden by an external user when compiled as a library
 */
__attribute__((weak)) int main(int argc, char **argv) {
  clover_main();
  return 0;
}
