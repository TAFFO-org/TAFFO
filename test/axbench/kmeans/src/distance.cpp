/*
 * distance.c
 * 
 * Created on: Sep 9, 2013
 * 			Author: Amir Yazdanbakhsh <a.yazdanbakhsh@gatech.edu>
 */


#include "distance.h"
#include <math.h>
#include <map>
#include <cstdio>

int count = 0;
#define MAX_COUNT 1200000

float euclideanDistance(RgbPixel* __attribute((annotate(ANNOTATION_RGBPIXEL))) p,
			Centroid* __attribute((annotate(ANNOTATION_CENTROID))) c1) {
	float __attribute((annotate("scalar()"))) r;
	float __attribute((annotate("scalar(range(1.0e-2,2.976608) final)"))) rr;

	r = 0;
	double __attribute((annotate("scalar()"))) r_tmp;

	/*
	double dataIn[6];
	dataIn[0] = RGBPIXEL_R(p, 0);
	dataIn[1] = RGBPIXEL_G(p, 0);
	dataIn[2] = RGBPIXEL_B(p, 0);
	dataIn[3] = CENTROID_R(c1, 0);
	dataIn[4] = CENTROID_G(c1, 0);
	dataIn[5] = CENTROID_B(c1, 0);
	*/

//#pragma parrot(input, "kmeans", [6]dataIn)

	r += (p->r - c1->r) * (p->r - c1->r);
	r += (p->g - c1->g) * (p->g - c1->g);
	r += (p->b - c1->b) * (p->b - c1->b);

	rr = r;

	r_tmp = sqrt(rr);

//#pragma parrot(output, "kmeans", <0.0; 1.0>r_tmp)

	// fprintf(stderr, "%f\n", r_tmp);

	return r_tmp;
}

int pickCluster(RgbPixel* __attribute((annotate(ANNOTATION_RGBPIXEL))) p,
		Centroid* __attribute((annotate(ANNOTATION_CENTROID))) c1) {
	float __attribute((annotate("scalar()"))) d1;

	d1 = euclideanDistance(p, c1);

	if (p->distance <= d1)
		return 0;

	return 1;
}

void assignCluster(RgbPixel* __attribute((annotate(ANNOTATION_RGBPIXEL))) p,
		   Clusters* __attribute((annotate(ANNOTATION_CLUSTER))) clusters) {
	int i = 0;

	float __attribute((annotate("errtarget('distance') scalar()"))) d;
	d = euclideanDistance(p, &clusters->centroids[i]);
	p->distance = d;

	for(i = 1; i < clusters->k; ++i) {
		d = euclideanDistance(p, &clusters->centroids[i]);
		if (d < p->distance) {
			p->cluster = i;
			p->distance = d;
		}
	}
}
