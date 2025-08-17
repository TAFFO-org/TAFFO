/// TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>

int main(int argc, char* argv[]) {
  int a __attribute__((annotate("scalar(disabled)")));
  float b __attribute((annotate("scalar(range(-2,2) disabled)")));
  a = 1;
  b = 1234;
  printf("Values Begin\n");
  printf("%f\n", a / (b * 2.0));
  printf("%f\n", (b * 2.0) / a);
  printf("Values End\n");
  return 0;
}
