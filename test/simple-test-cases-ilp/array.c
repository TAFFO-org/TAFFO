#include <stdio.h>
#define ELEMENTS 10

int main(){
    double array __attribute((annotate("scalar(range(7, 2000))")));
    double sum;

    scanf("%f",&array);
        //load, process, store
    array = array;
    
    //double x = array[1] + array[2];

    printf("%f", array);
}