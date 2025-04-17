/// TAFFO_TEST_ARGS -disable-vra -lm
#include <math.h>
#include <stdio.h>

float __attribute((annotate("range -20000 20000"))) global = 33.333;

float fun(float x, float y) {
  float local;
  local = x * y + global;
  global++;
  return local;
}

int funInt(float x, float y) {
  int local;
  local = x * y + global;
  global *= 1.098;
  return local;
}

int main() {
  float a = 10.2049;
  float __attribute((annotate("range -40000 40000"))) b = 10.1024;
  int c = 2;

  a = fun(b, a);
  printf("%f\n", a);

  a = fun(a, b);
  printf("%f\n", a);

  a = fun(b, b);
  printf("%f\n", a);

  a = fun(a, a);
  printf("%f\n", a);

  // ------------------ //

  a = a / 4000;

  b = fun(b, b);
  printf("%f\n", b);

  b = fun(a, b);
  printf("%f\n", b);

  b = fun(b, a);
  printf("%f\n", b);

  b = fun(a, a);
  printf("%f\n", b);

  // ----------------- //

  b = a / 4096;

  c = fun(b, b);
  printf("%d\n", c);

  c = fun(b, b);
  printf("%d\n", c);

  c = fun(b, a);
  printf("%d\n", c);

  c = fun(a, b);
  printf("%d\n", c);

  // ------------------ //

  printf("*******************\n");

  a = 10.05;

  a = funInt(b, b);
  printf("%f\n", a);

  a = funInt(b, a);
  printf("%f\n", a);

  a = funInt(a, b);
  printf("%f\n", a);

  a = funInt(a, a);
  printf("%f\n", a);

  a = sqrt(b);
  b = exp(a * 9.99);

  b = funInt(a, b);
  printf("%f\n", b);

  b = funInt(b, a);
  printf("%f\n", b);

  b = funInt(a, a);
  printf("%f\n", b);

  b = funInt(b, b);
  printf("%f\n", b);

  b = a;

  c = funInt(b, b);
  printf("%d\n", c);

  c = funInt(a, b);
  printf("%d\n", c);

  c = funInt(a, a);
  printf("%d\n", c);

  c = funInt(b, a);
  printf("%d\n", c);

  printf("-------------------\n");
  return 0;
}
