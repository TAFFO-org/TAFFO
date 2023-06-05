#include <stdio.h>
double calledSum(double a, double b);
int main(){
    float a __attribute((annotate("target('a') scalar(range(1,25) error(0.8))")));
    float b __attribute((annotate("target('b') scalar(range(1,2) error(0.9))")));
    float c __attribute((annotate("target('c') ")));

    scanf("%f", &a);
    scanf("%f", &b);

    c = exp((a*b)/a);
    float d=0;

    if (a == b) {
        d=c;
        printf("%f, %d", c, 2);
    }else{
        d=a;
        printf("%d, %f", 3, c);
    }

    printf("%d", d);
    printf("%d", calledSum(b, a));

    return 0;
}

double calledSum(double a, double b){
    return a+b;
}