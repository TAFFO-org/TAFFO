#include <stdio.h>
double a __attribute((annotate("scalar(range(1, 2))")));
double b __attribute((annotate("scalar(range(102400, 102400))")));
double *pointer;
double result;

double resolvePointer(double * pointing);

int main(){

    pointer = &a;
    //We are forcing the previous assignment, little clang
    printf("pointer: %x\n", pointer);

    scanf("%lf", pointer);
    //Incrementing the value pointed by the pointer i.e. incrementing a
    *pointer +=b;

    printf("a: %f, pointerref: %f\n", a, *pointer);

    result = resolvePointer(pointer);

    return 0;
}

double __attribute((annotate("scalar(range(1, 2))"))) resolvePointer(double * pointing){
    return *pointing;
}
