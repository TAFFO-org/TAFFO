/// TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>

#define PI 3.14159265358979323846264338

double cosTable[] = {1.0, 0.9999995000000417, 0.9999980000006666, 0.999995500003375, 0.9999920000106667};

double normalizeRadiantForCosine(double angle) {
  if (angle < 0)
    angle = -angle;
  while (angle >= 2 * PI)
    angle -= 2 * PI;
  return angle;
}

double cos2(double angle) {
  double a = normalizeRadiantForCosine(angle);      // [0, 2π)
  int index = (int)(a * (5.0 / (2*PI)) + 0.5);      // round to nearest bin
  if (index > 4) index = 4;                         // clamp upper edge
  return cosTable[index];
}

int main() {
  double angle;
  scanf("%lf", &angle);
  double taffo_angle __attribute__((annotate("scalar(range(0, 100))"))) = angle;
  printf("Values Begin\n");
  printf("%lf\n", cos2(taffo_angle));
  printf("Values End\n");
  return 0;
}
