///TAFFO_TEST_ARGS -posit -Xvra -propagate-all -Xdta -totalbits=8 -Xdta -maxtotalbits=20

// Manual test; please verify that the generated llvm-ir uses Posit16

#include <stdio.h>
#include <math.h>

float __attribute__((annotate("scalar()"))) x = INFINITY;

int main(int argc, char *argv[])
{
    printf("%f\n", x);
    return 0;
}