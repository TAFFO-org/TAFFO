///TAFFO_TEST_ARGS -Xvra -propagate-all
int main()
{
    float out;
    #pragma clang attribute push( __attribute__((annotate("scalar(range(-3000, 3000) final)"))) , apply_to = variable)
    float a=10;
    float b;
    #pragma clang attribute pop
    
    #pragma clang attribute push( __attribute__((annotate("scalar(range(-255, 255) final)"))) , apply_to = variable)
    float c = 2.1024;
    float d;
    #pragma clang attribute pop
    
    b = a * 0.21024;
    b /= 2;
    c /= 2;
    d = b + c;
    
    out = d;
    printf("%f\n",out);
    
    return 0;
}
