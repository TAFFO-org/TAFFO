#include <stdio.h>
#include <math.h>




int main(){

double __attribute__((annotate("target('x') scalar(range(-1, 1))"))) z_c = -1;
double __attribute__((annotate("target('y') scalar(range(-1, 1))"))) z = -1 ;

for (int i = 0; i < 160; i++){
    z = z_c + i*0.0125;

    printf("%f\n", cos(z));
}



}