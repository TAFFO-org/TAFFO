#define fptype double

#include <stdio.h>
#include <stdlib.h>
#include <math.h>



fptype CNDF(fptype __attribute((annotate("scalar()"))) InputX) {

    fptype __attribute((annotate("scalar()"))) expValues;


    expValues = exp(-0.5f * (InputX));

    return expValues;
}


int main() {
	fptype __attribute((annotate("target('start') scalar(range(-100, 100))"))) input;
    CNDF(input);
    return 0;
}