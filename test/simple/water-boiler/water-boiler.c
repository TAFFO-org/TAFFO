/// TAFFO_TEST_ARGS -Xvra -unroll=30
/* This program reads a sequence of pressure values in the circuit of a
 * water boiler, measured at regular intervals. The sequence is terminated by
 * the first negative number.
 *   The output of the program is the average of the various pressure values and
 * the average of the pressure peaks.
 *   A pressure value p1 is a peak (or local maxima) if it is followed by another
 * value p2 which is lesser than p1, and if it is preceded by another value p0
 * which is also lesser than p1.
 *   First and last value of the value sequence are not considered peaks. If no
 * values are inserted, the program prints a dash character ('-') in lieu of
 * the two averages. If no peaks are detected, it prints a dash only for the
 * peaks average. */

#include <stdio.h>

int main(int argc, char* argv[]) {
  float p1 __attribute__((annotate("scalar()")));
  float p2 __attribute__((annotate("scalar()")));
  float p3 __attribute__((annotate("scalar()")));
  float sPeak __attribute__((annotate("scalar() target('sPeak')")));
  float sAll __attribute__((annotate("scalar() target('sAll')")));
  int cPeak, cAll;

  p3 = p2 = p1 = -1;
  sPeak = sAll = 0;
  cPeak = cAll = 0;

  float tmp __attribute__((annotate("scalar(range(-3000, 3000) disabled)")));
  scanf("%f", &tmp);
  p3 = tmp;

  while (p3 > 0) {
    if (p3 > 0 && p2 > 0 && p1 > 0)
      if (p3 < p2 && p1 < p2) {
        sPeak += p2;
        cPeak++;
      }
    sAll += p3;
    cAll++;
    p1 = p2;
    p2 = p3;

    scanf("%f", &tmp);
    p3 = tmp;
  }

  printf("Values Begin\n");
  if (cPeak > 0)
    printf("%f\n", sPeak / cPeak);
  else
    printf("-1\n");
  if (cAll > 0)
    printf("%f\n", sAll / cAll);
  else
    printf("-1\n");
  printf("Values End\n");
  return 0;
}
