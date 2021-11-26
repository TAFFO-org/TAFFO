///TAFFO_TEST_ARGS -disable-vra
#include <stdio.h>
#include <math.h>

float hello(__attribute__((annotate("range -200 200"))) float abc) __attribute__((always_inline)) {
  return abc + (float)M_PI;
}


int main(int argc, char *argv[]) {
	__attribute__((annotate("range -200 200"))) float test = 123.0;
	test = hello(test);
	printf("%a\n", test);
	return 0;
}

