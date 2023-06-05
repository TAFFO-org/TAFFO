#include <stdio.h>

#define FPDATATYPE float
FPDATATYPE result __attribute((annotate("target('x') scalar(range(8,39))")));

int main(){
    FPDATATYPE x __attribute((annotate("target('x') scalar(range(1,25))")));
    FPDATATYPE y __attribute((annotate("target('y') scalar(range(7,14))")));

    scanf("%lf", &x);
    scanf("%lf", &y);

    result = x + y;

    printf("%lf", result);

    return 0;
}