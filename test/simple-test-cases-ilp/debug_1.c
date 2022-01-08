#define fptype double

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

fptype __attribute((annotate("scalar(range(-100, 100))"))) input;

fptype CNDF(fptype __attribute((annotate("scalar()"))) InputX) {

    fptype __attribute((annotate("scalar()"))) expValues;


    expValues = exp(-0.5f * (InputX));

    return expValues;
}


int main() {
    CNDF(input);
    return 0;
}