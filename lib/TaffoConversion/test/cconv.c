#include <stdio.h>
#include <stdlib.h>


#define N 10
#define FLOAT_TYPE float

void arrayLoad(FLOAT_TYPE array[], int elem){
    for(int i=0; i<elem; i++){
        scanf("%f", &array[i]);
    }

}

void arrayInit(FLOAT_TYPE array[], int elem, FLOAT_TYPE value){
    for(int i=0; i<elem; i++){
        array[i] = value;
    }

}


void printArray(FLOAT_TYPE array[], int elem){
    for(int i=0; i<elem; i++){
        printf("%f, ", array[i]);
    }

}


void cconv(FLOAT_TYPE xn[], FLOAT_TYPE h[], FLOAT_TYPE res[], int elem){
    for(int n=0; n<elem; n++){
        for(int i=0; i<elem;i++){
            res[n] += h[i] * xn[((n-i)+N)%N];
        }
    }
}

int main(){
    FLOAT_TYPE xn[N] __attribute((annotate("scalar(range(1, 20))")));
    FLOAT_TYPE h[N] __attribute((annotate("scalar(range(1, 20))")));

    FLOAT_TYPE res[N];

    arrayLoad(xn, N);
    arrayLoad(h, N);

    arrayInit(res, N, 0);

    cconv(xn, h, res, N);

    printArray(res, N);

}