//
// Created by nicola on 25/05/19.
//

#ifndef JPEG_DECODER_IDCT_H
#define JPEG_DECODER_IDCT_H

//#include <math.h>
#include "cosineTable.h"

#define PI 3.14159265358979323846264338
#define SQRT2 1.41421356237309504880168872420969807856967187537694807317667973799

double cos2(int angle);

void idct2(const int *table, double *output) {

    double input[64] __attribute((annotate("target('input') scalar(range(-1024, 1024) error(1e-8))")));

    int u, v, x, y;

    double z __attribute((annotate("target('z') scalar(range(-1024, 1024))"))) = 0.0; //(-622 624)

    for (v = 0; v < 8; v++) {
        for (u = 0; u < 8; u++) {
            input[v * 8 + u] = table[v * 8 + u];
            if (u == 0) input[v * 8 + u] *= 1 / SQRT2;
            if (v == 0) input[v * 8 + u] *= 1 / SQRT2;
        }
    }


    for (y = 0; y < 8; y++) {
        for (x = 0; x < 8; x++) {
            z = 0;
            for (v = 0; v < 8; v++) {
                for (u = 0; u < 8; u++) {
                    int xu = (2 * x + 1) * u * 196; //This is not magic but PI / 16 *1000 (the last operand is for the lookup table)
                    int yv = (2 * y + 1) * v * 196;
                    double addend __attribute((annotate("target('addend') scalar(range(-100, 100))")));
                    addend = input[v * 8 + u]
                         * cos2((int)(xu))
                         * cos2((int)(yv));

                    z+=addend;



                }
            }
            output[x + y * 8] = z / 4.0;

        }
    }

}


 inline double __attribute((annotate("scalar(final range(-1, 1))")))cos2(int angle) {
    if (angle < 0) angle = -angle;
    //while (angle >= 2 * PI)angle -= 2 * PI;
    //int index = (int) ((angle) * 1000 );
    //while(index>=6282)index -= 6282;

    return cosTable[angle%(6282)];
}


#endif //JPEG_DECODER_IDCT_H
