///TAFFO_TEST_ARGS -Xvra -propagate-all -Xvra -unroll=10
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

int main(int argc, char *argv[]) {
  float p1 __attribute((annotate("scalar()")));
  float p2 __attribute((annotate("scalar()")));
  float p3 __attribute((annotate("scalar(range(-3000, 3000) disabled final)")));
  float sPeak __attribute((annotate("scalar()")));
  float sAll __attribute((annotate("scalar()")));
  int cPeak, cAll;
  
  p3 = p2 = p1 = -1;
  sPeak = sAll = 0;
  cPeak = cAll = 0;
  
  scanf("%f",&p3);
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
    scanf("%f",&p3);
  }
  
  printf("Peak average: ");
  if (cPeak > 0)
    printf("%f\n", sPeak / cPeak);
  else
    printf("-\n");
  printf("Global average: ");
  if (cAll > 0)
    printf("%f\n", sAll / cAll);
  else
    printf("-\n");
    
  return 0;
}
