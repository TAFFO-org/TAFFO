///TAFFO_TEST_ARGS -posit -Xvra -propagate-all

#include <stdio.h>
#include <inttypes.h>

int main() {
    float __attribute__((annotate("scalar(range(-100.0, 100.0))"))) x;

    double tmp;
    scanf("%lf", &tmp);
    x = tmp;

    printf("Int32: %"PRId32"\n", (int32_t)x);
    printf("Int64: %"PRId64"\n", (int64_t)x);

    float f = x;
    printf("Single precision: %.17g\n", f);

    double d = x;
    printf("Double precision: %.17g\n", d);

    int64_t i = x;
    x = i;
    printf("Integer to Posit: %.17g\n", x);
}
