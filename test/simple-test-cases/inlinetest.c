///TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>


float hello(__attribute__((annotate("scalar()"))) float abc) __attribute__((always_inline)) {
  return abc + (float)5.0;
}


int main(int argc, char *argv[]) {
	__attribute__((annotate("scalar()"))) float test = 123.0;
	test = hello(test);
	printf("%a\n", test);
	return 0;
}

