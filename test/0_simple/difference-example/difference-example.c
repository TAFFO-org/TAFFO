/// TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>

double test(void) {
  double a __attribute__((annotate("scalar()")));
  double b = 98.0, c = 77.0;
  b = b - c;
  a = b * 2.0;
  return a;
}

int main(int argc, char* argv[]) {
  printf("Values Begin\n");
  printf("%lf\n", test());
  printf("Values End\n");
  return 0;
}
