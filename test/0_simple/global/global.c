/// TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>

double vec[10] __attribute__((annotate("scalar()")));
double scal __attribute__((annotate("scalar()")));

int main(int argc, char* argv[]) {
  for (__attribute__((annotate("scalar(range(0, 10))"))) int i = 0; i < 10; i++)
    vec[i] = i / 2.0;
  scal = 5.0;

  printf("Values Begin\n");
  for (int i = 0; i < 10; i++)
    printf("%f\n", vec[i] * scal);
  printf("Values End\n");
  return 0;
}
