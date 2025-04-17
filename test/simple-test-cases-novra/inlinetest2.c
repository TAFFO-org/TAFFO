/// TAFFO_TEST_ARGS -disable-vra
#include <math.h>
#include <stdio.h>

float hello(__attribute__((annotate("range -200 200"))) float* abc) __attribute__((always_inline)) {
  abc[5] += (float) M_PI;
}

int main(int argc, char* argv[]) {
  __attribute__((annotate("range -200 200"))) float test[10];
  for (int i = 0; i < 10; i++)
    test[i] = 123.0;
  hello(test);
  printf("%a\n", test[5]);
  return 0;
}
