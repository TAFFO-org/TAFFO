#include "IDCT.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(){
    srand(time(0));
    
    int input[64];
    
    for(int i=0; i<64; i++){
        input[i]=rand()%10;
    }
    
    double output[64] __attribute((annotate("scalar(range(-1024, 1024))")));
    
    clock_t t;
    t = clock();
    
    for(int j=0; j<100000; j++)
        idct2(input, output);
    
    t = clock() - t;
    double time_taken = ((double) t) / CLOCKS_PER_SEC; // in seconds
    
    for(int i=0; i<64; i++){
        double a =output[i];
        printf("%lf \n", a);
    }
    
    printf("Time taken: %f", time_taken);
}
