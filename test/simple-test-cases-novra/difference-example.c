///TAFFO_TEST_ARGS -disable-vra
#include <stdio.h>

double test(void)
{
	double a __attribute((annotate("range -100 100")));
	double b = 98.0, c = 77.0;
	b = b - c;
	a = b * 2.0;
	return a;
}

int main(int argc, char *argv[])
{
	printf("%lf\n", test());
	return 0;
}


