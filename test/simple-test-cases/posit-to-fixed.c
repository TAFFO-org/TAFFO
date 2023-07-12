///TAFFO_TEST_ARGS -posit -Xvra -propagate-all

#include <stdio.h>

int main() {
    float __attribute__((annotate("scalar(range(-10, 10))"))) x;
    float __attribute__((annotate("scalar(type(64 2))"))) y;
    float tmp;
    scanf("%f", &tmp);
    x = tmp;
    y = x;
    printf("%f\n", y);
    y = tmp;
    printf("%f\n", y);
    x = y;
    printf("%f\n", x);
    return 0;
}
