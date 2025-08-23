/// TAFFO_TEST_ARGS
#include <stdio.h>

float glob __attribute__((annotate("scalar()")));

float* fun(void) __attribute__((annotate("scalar()"))) { return &glob; }

int main() {
  float* x __attribute__((annotate("target('x') scalar()"))) = fun();
  float t __attribute__((annotate("scalar(range(-10, 10) disabled)")));
  scanf("%f", &t);
  *x = t;
  printf("Values Begin\n");
  printf("%f\n", *x);
  printf("Values End\n");
  return 0;
}
