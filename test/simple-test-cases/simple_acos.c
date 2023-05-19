///TAFFO_TEST_ARGS -fixm -Xvra -propagate-all -lm

#include <math.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  float tmp __attribute((annotate("target('a') scalar(range(-1, 1))"))) = -1;

  for (int i = 0; i < 20; i++) {
    float __attribute((annotate("scalar()"))) p = acos(tmp);
    printf("%f) %f\n", tmp, p);
    tmp += 0.1f;
  }

  return 0;
}
