/// TAFFO_TEST_ARGS -disable-vra
#include <math.h>
#include <stdio.h>

float hello(__attribute__((annotate("range -200 200"))) float abc) __attribute__((always_inline)) {
  return abc + (float) M_PI;
}

int main(int argc, char* argv[]) {
  __attribute__((annotate("range -200 200"))) float test = 123.0;
  test = hello(test);
  printf("%a\n", test);
  return 0;
}
