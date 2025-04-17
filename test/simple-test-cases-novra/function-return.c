/// TAFFO_TEST_ARGS -disable-vra
#include <math.h>
#include <stdio.h>

float fun(void) __attribute((annotate("scalar(range(-10, 10))"))) {
  float t;
  scanf("%f", &t);
  return t;
}

int main() {
  float x = fun();
  printf("%f\n", x);
  return 0;
}
