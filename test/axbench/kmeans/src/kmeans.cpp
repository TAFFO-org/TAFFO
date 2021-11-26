/*
 * kmeans.c
 * 
 * Created on: Sep 9, 2013
 * 			Author: Amir Yazdanbakhsh <a.yazdanbakhsh@gatech.edu>
 */


#include <stdio.h>
#include <string>
#include "rgbimage.h"
#include "segmentation.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "benchmark.hpp"

int main (int argc, const char* argv[]) {

	RgbImage __attribute((annotate(ANNOTATION_RGBIMAGE))) srcImage;
	Clusters __attribute((annotate("target('clusters') " ANNOTATION_CLUSTER))) clusters;

	initRgbImage(&srcImage);

	std::string inImageName  = argv[1];
	std::string outImageName = argv[2];

	loadRgbImage(inImageName.c_str(), &srcImage, 256);

	initClusters(&clusters, 6, 1);

	AxBenchTimer timer;
	segmentImage(&srcImage, &clusters, 1);
	uint64_t kernel_time = timer.nanosecondsSinceInit();

	std::cout << "kernel time = " << ((double)kernel_time) / 1000000000.0 << " s" << std::endl;

	saveRgbImage(&srcImage, outImageName.c_str(), 255);


	freeRgbImage(&srcImage);
	freeClusters(&clusters);
	return 0;
}

