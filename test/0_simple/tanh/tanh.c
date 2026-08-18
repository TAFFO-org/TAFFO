#include <math.h>
#include <stdio.h>

int main() {
  int selector;
  scanf("%d", &selector);

  float __attribute__((annotate("target('main') scalar(range(-2, 1))"))) x;

  if (selector)
    x = -2.0f;
  else
    x = 1.0f;

  float __attribute__((annotate("scalar()"))) y = tanhf(x);

  printf("Values Begin\n");
  printf("%.10f\n", y);
  printf("Values End\n");

  return 0;
}
