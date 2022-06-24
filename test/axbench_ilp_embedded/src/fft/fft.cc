#include <cstdio>
#include <cmath>
#include <time.h>
#include "benchmark.hpp"

#define ANNOTATION_RANGE_N "scalar(range(1,2048) final disabled)"
#define ANNOTATION_COMPLEX(R1,R2) "struct[scalar(" R1 "),scalar(" R2 ")]"
#define ANNOTATION_COMPLEX_RANGE ANNOTATION_COMPLEX("range(-10000, 10000) final", "range(-10000, 10000) final")

#define PI 3.1415926535897931

#if 1
typedef struct {
   float real;
   float imag;
} Complex;
#endif

static int* indices;
static Complex* __attribute((annotate("target('x') " ANNOTATION_COMPLEX_RANGE))) x;
static Complex* __attribute((annotate("target('f') " ANNOTATION_COMPLEX(,)))) f;


static void fftSinCos(float __attribute((annotate("scalar()"))) x,
	       float* __attribute((annotate("scalar()"))) s,
	       float* __attribute((annotate("scalar()"))) c) {
    *s = sinf(-2 * PI * x);
    *c = cosf(-2 * PI * x);
}

static void calcFftIndices(int K, int* indices)
{
	int i, j;
	int N;

	N = (int)log2f(K) ;

	indices[0] = 0 ;
	indices[1 << 0] = 1 << (N - (0 + 1)) ;
	for (i = 1; i < N; ++i)
	{
		for(j = (1 << i) ; j < (1 << (i + 1)); ++j)
		{
			indices[j] = indices[j - (1 << i)] + (1 << (N - (i + 1))) ;
		}
	}
}

static void radix2DitCooleyTykeyFft(int K,
			     int* indices __attribute((annotate(ANNOTATION_RANGE_N))),
			     Complex* x __attribute((annotate("errtarget('x') " ANNOTATION_COMPLEX(,)))),
			     Complex* f __attribute((annotate("errtarget('f') " ANNOTATION_COMPLEX(,)))))
{
  /* This FFT implementation is buggy
   * x[0] should be < x[all i != 0] because the input is all positive, except it isn't
   * The actual maximum value is the integration of all values times 4 for some reason */
	calcFftIndices(K, indices) ;

	int step ;
	float __attribute((annotate("scalar()"))) arg ;
	int eI ;
	int oI ;

	float __attribute((annotate("scalar()"))) fftSin ;
	float __attribute((annotate("scalar()"))) fftCos ;

	Complex __attribute((annotate(ANNOTATION_COMPLEX_RANGE))) t;
	int i ;
	int __attribute((annotate(ANNOTATION_RANGE_N))) N ;
	int j ;
	int __attribute((annotate(ANNOTATION_RANGE_N))) k ;

/*
	double dataIn[1];
	double dataOut[2];
*/
	for(i = 0, N = 1 << (i + 1); N <= K ; i++, N = 1 << (i + 1))
	{
		for(j = 0 ; j < K ; j += N)
		{
			step = N >> 1 ;
			for(k = 0; k < step ; k++)
			{
				arg = (float)k / N ;
				eI = j + k ;
				oI = j + step + k ;
/*
				dataIn[0] = arg;

#pragma parrot(input, "fft", [1]dataIn)
*/
				fftSinCos(arg, &fftSin, &fftCos);
/*
				dataOut[0] = fftSin;
				dataOut[1] = fftCos;

#pragma parrot(output, "fft", [2]<0.0; 2.0>dataOut)

				fftSin = dataOut[0];
				fftCos = dataOut[1];
*/
				// Non-approximate
				t =  x[indices[eI]] ;
				x[indices[eI]].real = t.real + (x[indices[oI]].real * fftCos - x[indices[eI]].imag * fftSin);
				x[indices[eI]].imag = t.imag + (x[indices[eI]].imag * fftCos + x[indices[oI]].real * fftSin);

				x[indices[oI]].real = t.real - (x[indices[oI]].real * fftCos - x[indices[eI]].imag * fftSin);
				x[indices[oI]].imag = t.imag - (x[indices[eI]].imag * fftCos + x[indices[oI]].real * fftSin);
			}
		}
	}

	for (int i = 0 ; i < K ; i++)
	{
		//f[i] = x[indices[i]] ;
		f[i].real = x[indices[i]].real;
		f[i].imag = x[indices[i]].imag;
	}
}

#ifndef BENCH_MAIN
#define BENCH_MAIN main
#endif
extern "C" int BENCH_MAIN(int argc, char* argv[])
{
	int i ;

	int __attribute((annotate("target('n') scalar(range(1,65536) final disabled)"))) n = 2048;

	// create the arrays
	x 		= (Complex*)malloc(n * sizeof (Complex));
	f 		= (Complex*)malloc(n * sizeof (Complex));
	indices = (int*)malloc(n * sizeof (int));

	if(x == NULL || f == NULL || indices == NULL)
	{
                printf("cannot allocate memory for the triangle coordinates!\n");
		return -1 ;
	}

	int K = n;

	for(i = 0;i < K ; i++)
	{
		x[i].real = i < (K / 100) ? 1.0 : 0.0;
		x[i].imag = 0 ;
	}

        {
	AxBenchTimer timer;

	radix2DitCooleyTykeyFft(K, indices, x, f) ;

	uint32_t time = timer.cyclesSinceReset();
        printf("kernel time = %u units \n", time);
	}

	for(i = 0;i < K ; i++)
	{
          printf("%f %f %d\n", f[i].real, f[i].imag, indices[i]);
	}
	
	free(x);
	free(f);
	free(indices);

	return 0 ;
}
