#include <stdio.h>
#include <math.h>




int main(){

double __attribute__((annotate("target('x') scalar(range(0, 1))"))) z_c = 0;
double __attribute__((annotate("target('y') scalar(range(0, 1))"))) z = 0 ;

for (int i = 0; i < 80; i++){
    z = z_c + i*0.0125;

    printf("%f\n", sin(z));
}



}