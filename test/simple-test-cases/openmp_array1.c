///TAFFO_TEST_ARGS -Xvra -propagate-all -fopenmp
#include <stdio.h>

#define MAX_N (100)


int main(int argc, char *argv[])
{
  float array[MAX_N] __attribute__((annotate("target('array') scalar(range(0,100) final)")));

  int i = 0;

  #pragma omp parallel for
  for (i = 0; i < MAX_N; i++) {
    array[i] = i * 1.0;
  }

  float result __attribute__((annotate("target('result') scalar(range(0,6000) final)"))) = 0;

  for (i = 0; i < MAX_N; i++) {
    result += array[i];
    printf("%d: %f\n", i, array[i]);
  }

  printf("result: %f\n", result);
}
