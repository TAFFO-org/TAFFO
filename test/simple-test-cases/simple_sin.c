///TAFFO_TEST_ARGS -fixm -Xvra -propagate-all -lm
#include <math.h>
#include <stdio.h>



int main(int argc, char *argv[])
{
  int n = 0;
  float tmp __attribute((annotate("target('a') scalar(range(-100, 100))")))=-100;

  for (int i = 0; i < 2000; i++){
  float p = sin(tmp);
	printf("%f\n",  p);
	tmp += 0.1;
  }
  return 0;
}

