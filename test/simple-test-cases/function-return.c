///TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>


float fun(void) __attribute((annotate("scalar(range(-10, 10))")))
{
  float t;
  scanf("%f", &t);
  return t;
}


int main()
{
  float x = fun();
  printf("%f\n", x);
  return 0;
}
