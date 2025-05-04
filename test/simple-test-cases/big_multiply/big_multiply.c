/// TAFFO_TEST_ARGS - lm
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double deconstify(double value) {
  asm volatile("" : : "r,m"(value) : "memory");
  return value;
}

void u_second_bigger() {
  double __attribute__((annotate("scalar(range(1234567000.9, 1234567000.9))"))) a = deconstify(123456.9);
  double __attribute__((annotate("scalar(range(1111111111.1, 1111111111.1))"))) c = deconstify(1111111111.1);
  double __attribute__((annotate("scalar() target('1')"))) d = a * c;
  printf("%lf\n", d);
}

void u_normal() {
  double __attribute__((annotate("scalar(range(2, 2))"))) a = deconstify(2);
  double __attribute__((annotate("scalar(range(1111111111.1, 1111111111.1))"))) c = deconstify(1111111111.1);
  double __attribute__((annotate("scalar(range(2222222222.2, 2222222222.2)) target('1')"))) d = a * c;
  printf("%lf\n", d);
}

void u_first_bigger() {
  double __attribute__((annotate("scalar(range(1234567000.9, 1234567000.9))"))) c = deconstify(123456.9);
  double __attribute__((annotate("scalar(range(1111111111.1, 1111111111.1))"))) a = deconstify(1111111111.1);
  double __attribute__((annotate("scalar() target('2')"))) d = a * c;
  printf("%lf\n", d);
}

void s_second_bigger() {
  double __attribute__((annotate("scalar(range(-1234567000.9, 1234567000.9))"))) a = deconstify(-123456.9);
  double __attribute__((annotate("scalar(range(-1111111111.1, 1111111111.1))"))) c = deconstify(-1111111111.1);
  double __attribute__((annotate("scalar() target('3')"))) d = a * c;
  printf("%lf\n", d);
}

void s_first_bigger() {
  double __attribute__((annotate("scalar(range(-1234567000.9, 1234567000.9))"))) c = deconstify(-123456.9);
  double __attribute__((annotate("scalar(range(-1111111111.1, 1111111111.1))"))) a = deconstify(-1111111111.1);
  double __attribute__((annotate("scalar() target('4')"))) d = a * c;
  printf("%lf\n", d);
}

void u_same() {
  double __attribute__((annotate("scalar(range(274877906976.9, 274877906976.9) type(64 20) )"))) c = deconstify(16.9);
  double __attribute__((annotate("scalar(range(51111111.1, 511111111.1) type(64 20))"))) a = deconstify(15.1);
  double __attribute__((annotate("scalar(type(64 10)) target('5')"))) d = a * c;
  printf("%lf\n", d);
}

void u_same_bigger_float() {
  double __attribute__((annotate("scalar(range(274877906976.9, 274877906976.9) type(64 55) )"))) c = deconstify(16.9);
  double __attribute__((annotate("scalar(range(51111111.1, 511111111.1) type(64 0))"))) a = deconstify(15.1);
  double __attribute__((annotate("scalar(type(64 40)) target('5')"))) d = a * c;
  printf("%lf\n", d);
}

int main(int argc, char* argv[]) {
  u_first_bigger();
  u_second_bigger();
  s_first_bigger();
  s_second_bigger();
  u_same();
  u_same_bigger_float();
  u_normal();

  return 0;
}
