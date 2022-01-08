#include <stdio.h>

#define FPDATATYPE float
FPDATATYPE result;
FPDATATYPE * result_ptr;
int main(){
    FPDATATYPE x __attribute((annotate("target('x') scalar(range(1,25))")));
    FPDATATYPE y __attribute((annotate("target('y') scalar(range(7,14))")));

    FPDATATYPE temp;

    scanf("%f", &temp);
    x=temp;
    scanf("%f", &temp);
    y=temp;
    //result_ptr = &x;

    result = x + y;

    printf("%f", result);

    result_ptr = &x;
    result_ptr = &y;

    return 0;
}