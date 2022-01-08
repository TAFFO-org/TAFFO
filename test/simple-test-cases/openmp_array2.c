///TAFFO_TEST_ARGS -Xvra -propagate-all -fopenmp
#include <omp.h>
#include <stdio.h>

#define MAX_N (100)

float compute_thread_result(int index, float private_multiplier)
{
  return index * private_multiplier;
}


int main(int argc, char *argv[])
{
  float result_container[MAX_N] __attribute((annotate("target('array') scalar(range(0,100) final)")));
  float multipliers_container[MAX_N] __attribute__((annotate("target('multipliers_container') scalar(range(0,1000) final)")));
  float result __attribute__((annotate("target('result') scalar(range(0,10000) final)"))) = 0;

  int i = 0;
  float private_multiplier __attribute__((annotate("target('private_multiplier') scalar(range(0,25) final)")));

  #pragma omp parallel for private(private_multiplier) num_threads(4) schedule(static)
  {
    for (i = 0; i < MAX_N; i++) {
      private_multiplier = 5.43;
      // Do computation on private variables
      private_multiplier *= omp_get_thread_num();
      multipliers_container[i] = private_multiplier;

      // Save in the shared variable accessed
      result_container[i] = compute_thread_result(i, multipliers_container[i]);
    }
  }


  for (i = 0; i < MAX_N; i++) {
    result += result_container[i];
  }

  printf("result: %f\n", result);
}
