#include <math.h>
#include <stdio.h>

float swish(float x) { return x / (1.0f + expf(-x)); }

int main() {
  int selector;
  scanf("%d", &selector);

  float __attribute__((annotate("target('main') scalar(range(-2, 1))"))) x;

  if (selector)
    x = -2.0f;
  else
    x = 1.0f;

  float __attribute__((annotate("scalar()"))) y = swish(x);

  printf("Values Begin\n");
  printf("%.10f\n", y);
  printf("Values End\n");

  return 0;
}
