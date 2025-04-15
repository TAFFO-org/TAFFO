#include <stdio.h>
#include <math.h>
double calledSum(double a, double b);
int main(){
    double a __attribute((annotate("scalar(range(1,25))")));
    double b __attribute((annotate("scalar(range(1,2))")));
    double c __attribute((annotate("")));

    scanf("%f", &a);
    scanf("%f", &b);

    c = exp((a*b)/a);
    int d=0;

    if (a == b) {
        d=c;
        printf("%f, %d", c, 2);
    }else{
        d=a;
        printf("%d, %f", 3, c);
    }

    printf("%d", d);
    printf("%d", calledSum(b, d));

    return 0;
}

double calledSum(double a, double b){
    return a+b;
}