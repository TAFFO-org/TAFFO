#include <stdio.h>
double a __attribute((annotate("target('a') scalar(range(0.1,1))")));
double b __attribute((annotate("target('b') scalar(range(0.1,1))")));
double c __attribute((annotate("target('c') scalar(range(0.1,10000))")));
double d __attribute((annotate("target('d') scalar(range(0.1,10000))")));
double e __attribute((annotate("target('e') scalar(range(0.1,2))")));
double f __attribute((annotate("target('f') scalar(range(0.1,20000))")));
double g __attribute((annotate("target('g') scalar(range(0.1,20000))")));

int main(){



    e=a+b;
    f=c+d;

    g=f*e;


}