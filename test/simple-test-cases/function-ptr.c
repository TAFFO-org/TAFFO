///TAFFO_TEST_ARGS
#include <stdio.h>
#include <stdlib.h>

float __attribute((annotate("scalar()"))) global = 3.333;

float** fun(float **x, float *y){
    float local;
    local = **x * *y + global;
    **x = 4.9876;
    printf("%f\n",local);
    return x;
}

int main() {
    float __attribute((annotate("target('a') scalar()"))) *a;

    float __attribute((annotate("scalar()"))) b = 10.10;
    float c = 2.2;

    a=&b; //TODO  bug if i42* != i42*

    float** __attribute((annotate("scalar()"))) k;

    float* __attribute((annotate("scalar()"))) mall = (float*)malloc(5 * (sizeof(float)+1));

    mall[0] = 0.1;
    *(mall+1) = 1.1;
    mall[2] = 2.2;

    for (int i=0;i<3;i++)
         printf("%f\n",*(mall+i));

    for (int i=0;i<5;i++)
        mall[i] = (float) (i) / (float) (i+1);

    for (int i=0;i<5;i++)
         printf("%f\n",mall[i]);

    printf("%f\n",*a);
    k = fun(&a,&b); //TODO  bug if i42** != float**
    printf("%f\n",*a);
    printf("%f\n",**k);


    printf("-------------------\n");
    return 0;
}
