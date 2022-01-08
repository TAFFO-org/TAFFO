///TAFFO_TEST_ARGS -fopenmp
#include <stdio.h>

#define NUM_THREADS (10)


int main(int argc, char *argv[])
{
  float result __attribute__((annotate("scalar(range(0,5000))"))) = 0.0;

  #pragma omp parallel reduction(+:result) num_threads(NUM_THREADS)
  result += 500.0;

  printf("result: %f\n", result);
}
