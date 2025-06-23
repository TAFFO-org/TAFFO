/// TAFFO_TEST_ARGS -lm
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double deconstify(double value) {
  asm volatile("" : : "r,m"(value) : "memory");
  return value;
}

int main(int argc, char* argv[]) {
  double __attribute__((annotate("scalar(range(1234567.9, 1234567.9))"))) a = deconstify(1234567.9);
  double __attribute__((annotate("scalar(range(72.0, 72.0))"))) b = deconstify(72.0);
  double __attribute__((annotate("scalar(range(11111111.1, 11111111.1))"))) c = deconstify(11111111.1);
  double __attribute__((annotate("scalar() target('d')"))) d = fma(a, b, c);
  printf("%lf %lf %lf %lf\n", a, b, c, d);
  return 0;
}
