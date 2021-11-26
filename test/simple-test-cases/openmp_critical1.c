///TAFFO_TEST_ARGS -Xvra -propagate-all -fopenmp
#include <stdio.h>

#define MAX_N (100)


int main(int argc, char *argv[])
{
  float result __attribute__((annotate("scalar(range(0,100))"))) = 0.0;

  int i = 0;

  #pragma omp parallel for
  for (i = 0; i < MAX_N; i++) {
    #pragma omp critical
    result += 1.0;
  }

  printf("result: %f\n", result);
}
