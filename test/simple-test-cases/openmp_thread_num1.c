///TAFFO_TEST_ARGS -fopenmp
#include <omp.h>
#include <stdio.h>

#define NUM_THREADS 4

int main(void)
{
  float container[NUM_THREADS] __attribute__((annotate("target('container') scalar(range(0,1) final)")));
  int i;

  #pragma omp parallel num_threads(NUM_THREADS)
	{
	  float x __attribute__((annotate("target('x') scalar(range(0,5) final)"))) = 0.333333;
	  int index = omp_get_thread_num();

		container[index] = x * index;
	}

  for (i = 0; i < NUM_THREADS; i++)
    printf("%f\n", container[i]);
}
