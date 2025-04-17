/// TAFFO_TEST_ARGS -disable-vra
#include <stdio.h>

#define MAX_N (30)

int main(int argc, char* argv[]) {
  float numbers[MAX_N] __attribute((annotate("range -3000 3000")));
  int n = 0;
  float tmp __attribute((annotate("scalar(disabled range(-3000, 3000))")));

  for (int i = 0; i < MAX_N; i++) {
    if (scanf("%f", &tmp) < 1)
      break;
    numbers[n++] = tmp;
  }

  float add __attribute((annotate("range -3000 3000"))) = 0.0;
  float sub __attribute((annotate("range -3000 3000"))) = 0.0;
  float div __attribute((annotate("range -3000 3000"))) = 1.0;
  float mul __attribute((annotate("range -3000 3000"))) = 1.0;

  for (int i = 0; i < n; i++) {
    add += numbers[i];
    sub -= numbers[i];
    if (numbers[i] != 0.0)
      div /= numbers[i];
    mul *= numbers[i];
  }

  printf("add: %f\nsub: %f\ndiv: %f\nmul: %f\n", add, sub, div, mul);
  return 0;
}
