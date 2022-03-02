#include <stdio.h>

int add(double __attribute__((annotate("target('k') scalar(range(-10, 10))"))) x){
    return sin(x) + cos(x);
}

int main(){

double __attribute__((annotate("target('x') scalar(range(-10, 10))"))) z_c = -10;
double __attribute__((annotate("target('y') scalar(range(-10, 10))"))) z = -10.0 ;

for (int i = 0; i < 160; i++){
    z = z_c + i*0.125;

    printf("%f\n", add(z));
}



}