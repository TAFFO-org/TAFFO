///TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>


int main(int argc, char *argv[])
{
  float tmp __attribute((annotate("target('tmp') scalar(range(-3000, 3000))"))) = argc;
  float tmp2 __attribute((annotate("target('tmp2') scalar(range(-3000, 3000))")));

  tmp2 = sin(tmp);

  printf("%f", tmp2);
  return 0;
}
