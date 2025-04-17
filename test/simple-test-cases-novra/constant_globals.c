/// TAFFO_TEST_ARGS -disable-vra
#include <stdio.h>

static float __attribute((annotate("range -255 255"))) k[5] = {1, 2, 3, 4, 5};

static float __attribute((annotate("range -255 255"))) kx[][3] = {
  {-1, -2, -1},
  {0,  0,  0 },
  {1,  2,  1 }
};

static float __attribute((annotate("range -255 255"))) ky[][3] = {
  {-1, 0, 1},
  {-2, 0, 2},
  {-1, 0, 1}
};

int main(int argc, char* argv[]) {
  for (int i = 0; i < 5; i++)
    printf("%f ", k[i]);
  printf("\n");

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      printf("(%f, %f) ", kx[i][j], ky[i][j]);
    printf("\n");
  }
  printf("\n");

  return 0;
}
