///TAFFO_TEST_ARGS
#include <stdio.h>
#include <math.h>


void function_1_2(float * __attribute((annotate("scalar()"))) x, int y)
{
  *x = (*x) * y;
}


void function_1_1(float * __attribute((annotate("scalar()"))) x, int y)
{
  function_1_2(x, y);
}


void function_2_2(float * __attribute((annotate("scalar()"))) x)
{
  *x = (*x) * (*x);
}


void function_2_1(float * __attribute((annotate("scalar()"))) y)
{
  float __attribute((annotate("scalar(range(0, 10) disabled)"))) x;
  float __attribute((annotate("scalar()"))) x2;
  scanf("%f", &x);
  x2 = x;
  function_2_2(&x2);
  *y += x2;
}

int main(int argc, char *argv[])
{
  float x __attribute((annotate("target('x') scalar()"))) = 5.0;
  int y = 2.0;
  function_1_1(&x, y);
  printf("%f\n", x);
  function_2_1(&x);
  printf("%f\n", x);
  return 0;
}

