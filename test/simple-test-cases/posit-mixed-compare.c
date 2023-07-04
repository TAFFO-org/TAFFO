///TAFFO_TEST_ARGS -posit -Xvra -propagate-all

#include <stdio.h>

int main()
{
  float __attribute__((annotate("scalar(range(0, 360) type(32 24))"))) a;
  float __attribute__((annotate("scalar(range(0, 360))"))) b;
  float tmp;
  scanf("%f", &tmp);
  a = tmp;
  scanf("%f", &tmp);
  b = tmp;

  printf("%d\n", a<b);
  return 0;
}
