///TAFFO_TEST_ARGS -posit -Xvra -propagate-all

#include <stdio.h>

float rad(float x)
{
  float __attribute__((annotate("scalar(range(0, 6.29) type(64 54))"))) rad = x * (3.14159265359f / 180.0f);
  return rad;
}

int main()
{
  float __attribute__((annotate("scalar(range(0, 360))"))) theta = 90.0f;
  float __attribute__((annotate("scalar(range(0, 6.29))"))) res;

  res = rad(theta);
  printf("%f\n", res);
  return 0;
}
