///TAFFO_TEST_ARGS 
#include "omp.h"
#include <stdio.h>

int main(void) {
	float a;
	
  	#pragma omp parallel
	{
		__attribute__((annotate("no_float"))) float x=0.333333;		
		a = x + omp_get_thread_num();
  		printf("thread %f\n", a);
	}
}
