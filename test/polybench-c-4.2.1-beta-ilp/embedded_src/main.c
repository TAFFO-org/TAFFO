#include <stdio.h>
#include "stm32f2xx_hal.h"
#include "bench_main.h"

long unsigned dt;

void main(void)
{
  printf("hello, welcome to the crazy taffo bench campaign\n");
  #include "bench_main.c.in"
  printf("halting\n");
}




