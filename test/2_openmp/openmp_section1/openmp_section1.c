/// TAFFO_TEST_ARGS -fopenmp
#include <stdio.h>

#define N (100)

int main(int argc, char* argv[]) {
  float first_section __attribute__((annotate("target('first_section') scalar(range(0,10))"))) = 1.0f;
  float second_section __attribute__((annotate("target('second_section') scalar(range(0,2000))"))) = 363;
  // float result
  //     __attribute__((annotate("target('result') scalar(range(0,2000)))"))) =
  //         0;
  //  TODO: restore the annotation and retest
  float result = 0;

#pragma omp parallel num_threads(4)
  {
#pragma omp sections
    {
#pragma omp section
      { first_section *= 2.1f; }
#pragma omp section
      second_section *= 5.4f;
    }
  }
  result = first_section + second_section;

  printf("Values Begin\n");
  printf("%f\n", result);
  printf("Values End\n");
}
