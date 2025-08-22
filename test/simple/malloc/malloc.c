/// TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
  double* test __attribute__((annotate("scalar(range(-1, 1))")));
  test = malloc(10 * sizeof(double));
  for (int i = 0; i < 10; i++) {
    double tmp;
    scanf("%lf", &tmp);
    test[i] = tmp;

    printf("Values Begin\n");
    printf("%f\n", test[i]);
    printf("Values End\n");
  }
  return 0;
}
