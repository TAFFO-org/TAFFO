#ifndef __COMPLEX_HPP__
#define __COMPLEX_HPP__

#define ANNOTATION_RANGE_N "scalar(range(1,4194304) final disabled)"
#define ANNOTATION_COMPLEX(R1,R2) "struct[scalar(" R1 "),scalar(" R2 ")]"
#define ANNOTATION_COMPLEX_RANGE ANNOTATION_COMPLEX("range(-10000, 10000) final", "range(-10000, 10000) final")

#define PI 3.1415926535897931

#if 1
typedef struct {
   float real;
   float imag;
} Complex;
#endif

void fftSinCos(float x, float* s, float* c);

#if 0
float abs(const Complex* x);
float arg(const Complex* x);
#endif

#endif
