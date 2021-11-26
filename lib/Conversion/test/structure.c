#include <stdio.h>
#define ELEMENTS 10
struct astruct{
    int a;
    double b;
};

struct astruct a __attribute((annotate("target('a') struct[scalar(range(-7,7)),scalar(range(-8,8))]")));
struct astruct b __attribute((annotate("target('a') struct[scalar(range(-7,7)),scalar(range(-8,8))]")));
struct astruct z[2] __attribute((annotate("target('z') struct[scalar(range(-7,7)),scalar(range(-8,8))]")));

int main(){

    float x __attribute((annotate("target('x') scalar(range(-25,25))")));
    scanf("%lf %lf", &a.b, &x);
    x++;

    //a.b = a.b + a.a;
    //z[0].b+=x;
    z[1].b++;

    for(int i=0; i<2;i++){
        z[i].b++;
    }

    printf("%lf", a.b);


}