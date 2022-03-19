/*
 * inversek2j.cpp
 * 
 *  Created on: Sep. 10 2013
 *			Author: Amir Yazdanbakhsh <yazdanbakhsh@wisc.edu>
 */

#include <iostream>
#include <cstdlib>
#include <fstream> 
#include <time.h>
#include <iomanip>
#include <string>
#include <cmath>
#include "data.hpp"
#include "benchmark.hpp"


#define PI 3.141592653589

#define ANNOTATION_RANGE "range(1e-6,1.5707963267948966192313216916397)"


static float  __attribute((annotate("scalar(range(0.5,0.5) error(1e-8))"))) l1 = 0.5 ;
static float  __attribute((annotate("scalar(range(0.5,0.5) error(1e-8))"))) l2 = 0.5 ;

static void forwardk2j(float  __attribute((annotate("scalar()"))) theta1,
		float  __attribute((annotate("scalar()"))) theta2,
		float* __attribute((annotate("scalar()"))) x,
		float* __attribute((annotate("scalar()"))) y) {
	*x = l1 * cos(theta1) + l2 * cos(theta1 + theta2) ;
	*y = l1 * sin(theta1) + l2 * sin(theta1 + theta2) ;
}

static void inversek2j(float __attribute((annotate("scalar()"))) x,
		float __attribute((annotate("scalar()"))) y,
		float* __attribute((annotate("scalar() errtarget('theta')"))) theta1,
		float* __attribute((annotate("scalar() errtarget('theta')"))) theta2) {

/*
	double dataIn[2];
	dataIn[0] = x;
	dataIn[1] = y;

	double dataOut[2];

#pragma parrot(input, "inversek2j", [2]dataIn)
*/
	float __attribute((annotate("scalar(range(0.5,4.934802) final)"))) sqtmp = (x * x + y * y);

	*theta2 = (float)acos(((x * x) + (y * y) - (l1 * l1) - (l2 * l2))/(2 * l1 * l2));
	*theta1 = (float)asin((y * (l1 + l2 * cos(*theta2)) - x * l2 * sin(*theta2))/sqtmp);

/*
	dataOut[0] = (*theta1);
	dataOut[1] = (*theta2);

#pragma parrot(output, "inversek2j", [2]<0.0; 2.0>dataOut)


	*theta1 = dataOut[0];
	*theta2 = dataOut[1];
*/
}


#ifndef BENCH_MAIN
#define BENCH_MAIN main
#endif
extern "C" int BENCH_MAIN(int argc, const char* argv[])
{
	int n;

  n = INVERSEK2J_DATA_SIZE;

	float* __attribute((annotate("target('t1t2xy') scalar()"))) t1t2xy = (float*)malloc(n * 2 * 2 * sizeof(float));

	if(t1t2xy == NULL)
	{
		std::cerr << "# Cannot allocate memory for the coordinates an angles!" << std::endl;
		return -1 ;
	}

	//srand (time(NULL));

	for (int i=0; i<n*2*2; i+=2*2) {
		float __attribute((annotate("scalar(" ANNOTATION_RANGE " error(1e-8) disabled)"))) theta1, __attribute((annotate("scalar(" ANNOTATION_RANGE " error(1e-8) disabled)"))) theta2;
    theta1 = inversek2j_data[i].x;
    theta2 = inversek2j_data[i].y;
		t1t2xy[i] = theta1;
		t1t2xy[i + 1] = theta2;
	}

	AxBenchTimer timer;

	int curr_index1 = 0;
	for(int i = 0 ; i < n * 2 * 2 ; i += 2 * 2)
	{
		forwardk2j(t1t2xy[i + 0], t1t2xy[i + 1], t1t2xy + (i + 2), t1t2xy + (i + 3));
	}

	for(int i = 0 ; i < n * 2 * 2 ; i += 2 * 2)
	{
		inversek2j(t1t2xy[i + 2], t1t2xy[i + 3], t1t2xy + (i + 0), t1t2xy + (i + 1));
	}

	uint64_t time = timer.cyclesSinceReset();
	std::cout << "kernel time = " << time << " cycles\n";

	for(int i = 0 ; i < n * 2 * 2 ; i += 2 * 2)
	{
		std::cout <<  t1t2xy[i+0] << "\t" << t1t2xy[i+1] << "\n";
	}

	free(t1t2xy) ; 
	t1t2xy = NULL ;

	return 0 ;
}
