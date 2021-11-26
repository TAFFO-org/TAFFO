///TAFFO_TEST_ARGS -disable-vra
#include <stdio.h>
#include <math.h>


float glob __attribute((annotate("scalar(range(-10, 10))")));


float *fun(void) __attribute((annotate("scalar(range(-10, 10))")))
{
  return &glob;
}


int main()
{
  float *x __attribute((annotate("scalar(range(-10, 10))"))) = fun();
  float t;
  scanf("%f", &t);
  *x = t;
  printf("%f\n", *x);
  return 0;
}
