///TAFFO_TEST_ARGS -Xvra -propagate-all -fopenmp
#include <stdio.h>

#define MAX_N (100)


int main(int argc, char *argv[])
{
  float array[MAX_N] __attribute__((annotate("scalar(range(0,100))")));

  int i = 0;

  #pragma omp parallel for
  for (i = 0; i < MAX_N; i++) {
    array[i] = i * 1.0;
  }

  float result __attribute__((annotate("scalar(range(0,5000))"))) = 0;

  #pragma omp parallel for
  for (i = 0; i < MAX_N; i++) {
    #pragma omp critical
    result += array[i];
  }

  printf("result: %f\n", result);
}
