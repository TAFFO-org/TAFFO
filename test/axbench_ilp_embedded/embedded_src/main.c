#include <stdio.h>
#include "bench_main.h"

void main(void)
{
  printf("Welcome to the TAFFO Axbench ILP benchmark campaign.\n");
  #include "bench_main.c.in"
  printf("Halting.\n");
}
