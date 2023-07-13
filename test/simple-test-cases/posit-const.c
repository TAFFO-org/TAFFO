///TAFFO_TEST_ARGS -posit -Xvra -propagate-all -Xdta -totalbits=8

#include <stdio.h>

float __attribute__((annotate("scalar()"))) g = 7.0;

void test(float nonconst) {
    float __attribute__((annotate("scalar()"))) x = 2.0;
    float __attribute__((annotate("scalar()"))) y = 3.0;
    printf("%f\n", x - (-y)); // fully folded back to double
    printf("%f\n", nonconst * (x + y)); // mixed constant and variable
}

int main() {
    test(3.0f);
    g++;
    printf("%f\n", g);
    return 0;
}
