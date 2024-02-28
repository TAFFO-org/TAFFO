///TAFFO_TEST_ARGS -posit -Xvra -propagate-all

#include <stdio.h>

float __attribute__((annotate("scalar(type(posit 8))"))) g = 3.141f;

float test(float nonconst) {
    return g + nonconst;
}

int main() {
    printf("%f\n", test(3.0f));
    return 0;
}
