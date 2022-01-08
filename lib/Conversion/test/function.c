///TAFFO_TEST_ARGS 
#include <stdio.h>
#include <math.h>

double  __attribute((annotate("range 7 20000"))) global = 33.333;

double fun(double x, double y){
    float local;
    local = x * y + global;
    //global++;
    return local;
}

int funInt(double x, double y){
    int local;
    local = x * y + global;
    global*=1.098;
    return local;
}

int main() {
    double a=10.2049;
    double __attribute((annotate("range -7 40000"))) b=10.1024;
    int c = 2;
    
    
    a = fun(b,a);
    printf("%f\n",a);
    
    
    
    // ------------------ //
    
    a = a/4000;


    b = fun(a,b);
    printf("%f\n",b);

    
        
    // ----------------- //
    
    b = a/4096;

    
    c = fun(b,a);
    printf("%d\n",c);
    
    
    // ------------------ //
    
    
    printf("*******************\n");
    
    a=10.05;


    
    a = funInt(b,a);
    printf("%f\n",a);
    
    
    a = sqrt(b);
    b = exp(a*9.99);
    
    b = funInt(a,b);
    printf("%f\n",b);
    
    b = a;

    
    c = funInt(a,b);
    printf("%d\n",c);
    

    printf("-------------------\n");
    return 0;
}
