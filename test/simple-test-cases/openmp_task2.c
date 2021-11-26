///TAFFO_TEST_ARGS -fopenmp
#include <stdio.h>

#define MAX_N (100)

void nested_task_invocation(int index)
{
  if (index > 0)
    nested_task_invocation(index-1);
  else
#pragma omp task
  {
    printf("result: %d\n", index);
  }
}

void compute_result(int index)
{
  nested_task_invocation(index);
}

int main(int argc, char *argv[])
{
  float array[MAX_N] __attribute__((annotate("target('array') scalar(range(0,1000) final)")));
  float result __attribute__((annotate("target('result') scalar(range(0,5000) final)"))) = 0;

  int i;

  #pragma omp parallel
  {
    #pragma omp single
    compute_result(10);
  }

  #pragma omp parallel for
  for (i = 0; i < MAX_N; i++) {
    array[i] = i * 1.0;
  }

  for (i = 0; i < MAX_N; i++) {
    result += array[i];
  }

  printf("result: %f\n", result);
}
