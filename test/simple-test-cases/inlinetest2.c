///TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>


float hello(__attribute__((annotate("scalar()"))) float *abc) __attribute__((always_inline)) {
  abc[5] += (float)5.0;
}


int main(int argc, char *argv[]) {
	__attribute__((annotate("scalar()"))) float test[10];
	for (int i=0; i<10; i++)
	  test[i] = 123.0;
	hello(test);
	printf("%a\n", test[5]);
	return 0;
}

