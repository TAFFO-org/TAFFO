///TAFFO_TEST_ARGS -Xvra -propagate-all -Xdta -useposit -lTaffoPosit
#include <stdio.h>

double deconstify(double value)
{
  asm volatile("" : : "r,m"(value) : "memory");
  return value;
}

float fun(int b) {
    float __attribute__((annotate("scalar()"))) x = deconstify(-1);
    if (b > 1) {
        x = b;
    }
    return x;
}

int main() {
    int i;
    scanf("%d", &i);
    float __attribute__((annotate("scalar()"))) x = fun(i);
    printf("%d\n", (int)x);
    return 0;
}
