//
// Created by nicola on 25/05/19.
//

#include <math.h>

#define PI 3.14159265358979323846264338
#define SQRT2 1.41421356237309504880168872420969807856967187537694807317667973799

double* idct(double* data)
{

    double * output = (malloc(sizeof(double) * 8 * 8));
    int u,v,x,y;

    /* iDCT */
    for (y=0; y<8; y++)
        for (x=0; x<8; x++)
        {
            double z = 0.0;

            for (v=0; v<8; v++)
                for (u=0; u<8; u++)
                {
                    double S __attribute((annotate("range -1024 1024"))); //(-1024 1024)
                    double q __attribute((annotate("range -600 600"))); //(-600 600)
                    double Cu __attribute((annotate("range 0.7 1"))); //(0.7 1)
                    double Cv __attribute((annotate("range 0.7 1"))); //(0.7 1)

                    if (u == 0) Cu = 1.0 / SQRT2; else Cu = 1.0;
                    if (v == 0) Cv = 1.0 / SQRT2; else Cv = 1.0;
                    S = data[v*8+u];

                    q = Cu * Cv * S *
                        cos((double)(2*x+1) * (double)u * PI/16.0) *
                        cos((double)(2*y+1) * (double)v * PI/16.0);

                    z += q;


                }

            z /= 4.0;

            output[x+y*8] = z;
        }
    return output;
}

int main(){
    double a[64];

    double read=0;

    scanf("%f", &read);

    for(int i=0; i<64; i++){
     a[i]=read;
    }

    idct(a);
}