#include <stdio.h>

#define FPDATATYPE float
FPDATATYPE result;
FPDATATYPE * result_ptr;
int main(){
    FPDATATYPE x __attribute((annotate("target('x') scalar(range(1,25))")));
    FPDATATYPE y __attribute((annotate("target('y') scalar(range(7,14))")));

    scanf("%lf", &x);
    scanf("%lf", &y);
    result_ptr = &result;

    *result_ptr = x + y;

    printf("%lf", *result_ptr);

    return 0;
}