///TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>

double vec[10] __attribute((annotate("scalar()")));
double scal __attribute((annotate("scalar()")));


int main(int argc, char *argv[])
{
  for (int i=0; i<10; i++) {
    vec[i] = i / 2.0;
  }
  scal = 5.0;
  for (int i=0; i<10; i++) {
    printf("%f\n", vec[i]*scal);
  }
  return 0;
}
