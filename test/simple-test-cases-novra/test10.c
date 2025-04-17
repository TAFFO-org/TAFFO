/// TAFFO_TEST_ARGS -disable-vra
#include <stdio.h>

int main(int argc, char* argv[]) {
  float __attribute__((annotate("range 0 3"))) a = 3;
  float __attribute__((annotate("range 0 4"))) b = 4;
  float c = a * b;
  printf("%f", c);
  return 0;
}
