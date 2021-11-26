///TAFFO_TEST_ARGS -disable-vra
#include <stdio.h>


typedef struct {
  float a;
  int b;
  float c;
} test;


int main(int argc, char *argv[])
{
  test __attribute__((annotate("struct[scalar(range(-3000, +3000)), void, scalar(range(-3, +3))]"))) z;
  double a, b, c;
  scanf("%lf%lf%lf", &a, &b, &c);
  z.a = a;
  z.b = b;
  z.c = c;
  printf("%a\n%d\n%a\n", z.a, z.b, z.c);
  return 0;
}
