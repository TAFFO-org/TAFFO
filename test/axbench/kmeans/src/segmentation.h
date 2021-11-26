/*
 * segmentation.h
 * 
 * Created on: Sep 9, 2013
 * 			Author: Amir Yazdanbakhsh <a.yazdanbakhsh@gatech.edu>
 */


#ifndef SEGMENTATION_H_
#define SEGMENTATION_H_

#include "rgbimage.h"

typedef struct {
   float r;
   float g;
   float b;
   int n;
} Centroid;

#define ANNOTATION_CENTROID "struct[scalar(type(signed 32 12)),scalar(type(signed 32 12)),scalar(type(signed 32 12)),scalar(disabled range(1,200000))]"
#define ANNOTATION_CLUSTER  "struct[void," ANNOTATION_CENTROID "]"

typedef struct {
   int k;
   Centroid* centroids;
} Clusters;

int initClusters(Clusters* clusters, int k, float scale);
void segmentImage(RgbImage* image, Clusters* clusters, int n);
void freeClusters(Clusters* clusters);

#endif /* SEGMENTATION_H_ */
