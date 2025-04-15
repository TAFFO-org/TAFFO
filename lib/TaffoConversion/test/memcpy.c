#include <stdio.h>
#include <stdlib.h>

float source[5] __attribute((annotate("scalar(range(1, 1000000))")));
float dest[5] __attribute((annotate("scalar(range(1, 1))")));

float * destPtr =dest;


struct astruct{
    int a, b;
};

struct astruct strutturina;
int unavariabile;

int main(){
    destPtr = dest;
    scanf("%f%f", source, &source[1]);
    memcpy((void*)destPtr, (void*)source, sizeof(float)*5);
    printf("%f%f", dest, &dest[1]);

    return 0;
}