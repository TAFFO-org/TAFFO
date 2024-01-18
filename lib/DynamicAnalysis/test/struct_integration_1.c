///TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>


typedef struct {
  float a;
  int b;
  float c;
} test;

test __attribute__((annotate("struct[scalar(range(-3000, +3000)),void,scalar(range(-3, +3))]"))) z;

void print_struct(test* x) {
  printf("%a\n%d\n%a\n", x->a, x->b, x->c);
}

int main(int argc, char *argv[])
{
  float a, b, c;
  scanf("%f%f%f", &a, &b, &c);
  z.a = a;
  z.b = b;
  z.c = c;
  print_struct(&z);
  return 0;
}


