#include <stdio.h>

float relu(float x) { return x > 0.0f ? x : 0.0f; }

int main() {
  int selector;
  scanf("%d", &selector);

  float __attribute__((annotate("target('main') scalar(range(-2, 3))"))) x;

  float __attribute__((annotate("target('main') scalar(range(0, 3))"))) u;

  if (selector) {
    x = -2.0f;
    u = 1.0f;
  }
  else {
    x = 3.0f;
    u = 3.0f;
  }

  float __attribute__((annotate("scalar()"))) y_signed = relu(x);

  float __attribute__((annotate("scalar()"))) y_nonnegative = relu(u);

  printf("Values Begin\n");
  printf("%.10f\n", y_signed);
  printf("%.10f\n", y_nonnegative);
  printf("Values End\n");

  return 0;
}
