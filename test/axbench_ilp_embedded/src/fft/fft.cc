#include <cstdio>
#include <iostream>
#include "fourier.hpp"
#include <fstream>
#include <time.h>
#include "benchmark.hpp"

static int* indices;
static Complex* __attribute((annotate("target('x') " ANNOTATION_COMPLEX_RANGE))) x;
static Complex* __attribute((annotate("target('f') " ANNOTATION_COMPLEX(,)))) f;

int main(int argc, char* argv[])
{
	int i ;

	int __attribute((annotate("target('n') scalar(range(1,65536) final disabled)"))) n = 2048;

	// create the arrays
	x 		= (Complex*)malloc(n * sizeof (Complex));
	f 		= (Complex*)malloc(n * sizeof (Complex));
	indices = (int*)malloc(n * sizeof (int));

	if(x == NULL || f == NULL || indices == NULL)
	{
		std::cout << "cannot allocate memory for the triangle coordinates!" << std::endl;
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

	uint64_t time = timer.cyclesSinceReset();
	std::cout << "kernel time = " << time << " cycles \n";
	}

	for(i = 0;i < K ; i++)
	{
	  std::cout << f[i].real << " " << f[i].imag << " " << indices[i] << std::endl;
	}
	
	free(x);
	free(f);
	free(indices);

	return 0 ;
}
