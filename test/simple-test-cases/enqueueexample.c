///TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>


float example(float a, float b, float c, float d)
{
  float x = 1.0;
  float y __attribute((annotate("scalar(range(-10, 10))"))) = 2.0;
  float z =3.0;

  return (x * (y / z)) * x;
}


int main(int argc, char *argv[])
{
  printf("%f\n", example(1, 2, 3, 4));
}

