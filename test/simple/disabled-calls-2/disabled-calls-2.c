/// TAFFO_TEST_ARGS
#include <stdio.h>

float test(float a) { return a * 2; }

/*
 * Note: there is a test case using a number outside the input range
 * below to purposefully cause an integer overflow, to make sure variable
 * b gets converted to fixed-point.
 */

int main(int argc, char* argv[]) {
  float a __attribute((annotate("target('a') scalar(range(-128, 128) disabled)")));
  scanf("%f", &a);
  float b __attribute((annotate("target('a') scalar()"))) = a * 2;

  printf("Values Begin\n");
  printf("%f\n", test(b));
  printf("Values End\n");
  return 0;
}
