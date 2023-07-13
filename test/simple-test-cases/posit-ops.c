///TAFFO_TEST_ARGS -posit -Xvra -propagate-all -Xdta -totalbits=8 -Xdta -minfractbits=0

#include <stdio.h>

int main(int argc, char *argv[])
{
  float __attribute__((annotate("scalar(range(-100.0, 100.0))"))) x;
  float __attribute__((annotate("scalar(range(-200.0, 200.0))"))) y;
  volatile float __attribute__((annotate("scalar()"))) z;

  float tmp;
  scanf("%f", &tmp);
  x = tmp;
  scanf("%f", &tmp);
  y = tmp;

  z = x + y;
  printf("Add: %f\n", z);

  z = x - y;
  printf("Sub: %f\n", z);

  z = x * y;
  printf("Mul: %f\n", z);

  z = x / y;
  printf("Div: %f\n", z);

  printf("Neg: %f\t%f\n", -x, -y);

  printf("%s is bigger\n", x >= y ? "1st" : "2nd");

  return 0;
}
