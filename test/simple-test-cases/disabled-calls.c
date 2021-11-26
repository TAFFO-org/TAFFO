///TAFFO_TEST_ARGS
#include <stdio.h>

float test(float a)
{
  return a*2;
}

int main(int argc, char *argv[])
{
  float a __attribute((annotate("target('a') scalar(range(-128, 128) disabled)")));
  scanf("%f", &a);
  printf("%f\n", test(a));
  return 0;
}

