///TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


#define ANNOTATION_RGBPIXEL         "struct[scalar(range(0,255)),scalar(range(0,255)),scalar(range(0,255)),void,scalar(range(0,1))]"
#define ANNOTATION_RGBIMAGE         "struct[void,void," ANNOTATION_RGBPIXEL "]"

typedef struct {
   float r;
   float g;
   float b;
   int cluster;
   float distance;
} RgbPixel;

typedef struct {
   int w;
   int h;
   RgbPixel** pixels;
} RgbImage;


int main (int argc, const char* argv[])
{
	RgbImage __attribute((annotate(ANNOTATION_RGBIMAGE))) image;

	image.w = 10;
	image.h = 20;

  // check for 'undef's around this malloc
	image.pixels = (RgbPixel**)malloc(image.h * sizeof(RgbPixel*));

	return 0;
}

