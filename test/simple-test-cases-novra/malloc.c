///TAFFO_TEST_ARGS -disable-vra
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
  float *test __attribute__((annotate("scalar(range(-500, 500)) backtracking(2)")));
  test = malloc(10 * sizeof(float));
  for (int i=0; i<10; i++) {
    float tmp;
    scanf("%f", &tmp);
    test[i] = tmp;
    printf("%a\n", test[i]);
  }
  return 0;
}


