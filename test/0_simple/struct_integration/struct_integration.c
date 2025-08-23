/// TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>

typedef struct {
  float a;
  int b;
  float c;
} test;

int main(int argc, char* argv[]) {
  test __attribute__((annotate("struct[scalar(range(-3000, +3000)),void,scalar(range(-3, +3))]"))) z;
  float a, b, c;
  scanf("%f%f%f", &a, &b, &c);
  z.a = a;
  z.b = b;
  z.c = c;

  printf("Values Begin\n");
  printf("%f\n%d\n%f\n", z.a, z.b, z.c);
  printf("Values End\n");
  return 0;
}
