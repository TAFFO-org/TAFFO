///TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>

#define PI 3.14159265358979323846264338

double cosTable[] ={1.0, 0.9999995000000417, 0.9999980000006666, 0.999995500003375, 0.9999920000106667};


double normalizeRadiantForCosine(double angle){
    if (angle<0) angle = - angle;
    while(angle>=2*PI)angle -= 2*PI;
    return angle;
}

double cos2(double angle){
    int index = (int) (normalizeRadiantForCosine(angle) *1000 + 0.5555);
    return cosTable[index];
}

int main(){
    double angle;
    scanf("%lf", &angle);

    double taffo_angle __attribute((annotate("scalar(range(-20, 20))")))= angle;

    printf("%lf\n", cos2(taffo_angle));

    return 0;
}
