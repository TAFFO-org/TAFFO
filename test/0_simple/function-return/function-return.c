/// TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>

float fun(void) __attribute__((annotate("scalar(range(-10, 10))"))) {
  float t;
  scanf("%f", &t);
  return t;
}

int main() {
  float x = fun();
  printf("Values Begin\n");
  printf("%f\n", x);
  printf("Values End\n");
  return 0;
}
