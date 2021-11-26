///TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>

int main(int argc, char *argv[])
{
  int a __attribute((annotate("scalar(disabled)")));
  float b __attribute((annotate("scalar(range(-2,2))")));
  a = 1;
  b = 2;
  printf("%f\n", a/(b*2.0));
  printf("%f\n", (b*2.0)/a);
  return 0;
}


