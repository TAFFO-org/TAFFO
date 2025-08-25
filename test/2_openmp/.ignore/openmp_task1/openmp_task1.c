/// TAFFO_TEST_ARGS -fopenmp
#include <stdio.h>

#define N (100)

int main(int argc, char* argv[]) {
  float first_task_var __attribute__((annotate("target('first_task_var') scalar(range(0,10))"))) = 1.0f;
  float second_task_var __attribute((annotate("target('second_task_var') scalar(range(0,2000))"))) = 363;
  float result __attribute((annotate("target('result') scalar(range(0,2000))"))) = 0.0;

#pragma omp parallel num_threads(4)
  {
#pragma omp single
    {
      first_task_var += 1.0;
#pragma omp task
      { first_task_var *= 2.1f; }
#pragma omp task
      second_task_var *= 5.4f;
    }
  }
  result = first_task_var + second_task_var;

  printf("Values Begin\n");
  printf("%f\n", result);
  printf("Values End\n");
}
