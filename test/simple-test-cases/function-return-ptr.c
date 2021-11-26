///TAFFO_TEST_ARGS
#include <stdio.h>


float glob __attribute((annotate("scalar()")));


float *fun(void) __attribute((annotate("scalar()")))
{
  return &glob;
}


int main()
{
  float *x __attribute((annotate("target('x') scalar()"))) = fun();
  float t __attribute((annotate("scalar(range(-10, 10) disabled)")));
  scanf("%f", &t);
  *x = t;
  printf("%f\n", *x);
  return 0;
}
