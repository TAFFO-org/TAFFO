#include <stdio.h>
#include <stdlib.h>

int main(){
    double a __attribute((annotate("target('a') scalar(range(0,25) error(0.8))")));

    //This is a simple "forward" phi
    for(int i=0; i<5; i++){
        if(i==0){
            a=5;
        }else{
            a=7;
        }

        printf("%lf\n", a);
    }


    //This is a simple "backward" phi
    //backward means that the value is used in the ir before it is allocated!
    double b __attribute((annotate("target('b') scalar(range(0,25) error(0.8))")));
    b=42;
    for(int i=0; i<5; i++){
        if(i==0){
            b=0;
        }
        printf("%lf\n", b);
        b++;
    }

}