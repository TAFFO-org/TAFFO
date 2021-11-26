/*
 * convolution.hpp
 *
 * Created on: Sep 9, 2013
 *          Author: Amir Yazdanbakhsh <a.yazdanbakhsh@gatech.edu>
 */

 #ifndef __CONVOLUTION_HPP__
 #define __CONVOLUTION_HPP__

#include "rgb_image.hpp"
#include <iostream>

float sobel(float w[][3]) ;

#define WINDOW(imagePtr, x, y, window) {			\
 	window[0][0] = imagePtr->pixels[ y - 1 ][ x - 1 ].r ;	\
 	window[0][1] = imagePtr->pixels[ y - 1 ][ x ].r ;	\
 	window[0][2] = imagePtr->pixels[ y - 1 ][ x + 1 ].r ;	\
\
	window[1][0] = imagePtr->pixels[ y ][ x - 1 ].r ;	\
 	window[1][1] = imagePtr->pixels[ y ][ x ].r ;	    	\
 	window[1][2] = imagePtr->pixels[ y ][ x + 1 ].r ;	\
\
	window[2][0] = imagePtr->pixels[ y + 1 ][ x - 1 ].r ;	\
 	window[2][1] = imagePtr->pixels[ y + 1 ][ x ].r ;	\
 	window[2][2] = imagePtr->pixels[ y + 1 ][ x + 1 ].r ;	\
}

#define HALF_WINDOW(imagePtr, x, y, window) {											\
	window[0][0] = (x == 0 || y == 0					) ? 0 : imagePtr->pixels[y - 1][x - 1].r;	\
	window[0][1] = (y == 0							) ? 0 : imagePtr->pixels[y - 1][x].r;		\
	window[0][2] = (x == imagePtr->width -1 || y == 0			) ? 0 : imagePtr->pixels[y - 1][x + 1].r;	\
\
	window[1][0] = (x == 0 							) ? 0 : imagePtr->pixels[y][x - 1].r;		\
	window[1][1] = 									imagePtr->pixels[y][x].r;		\
	window[1][2] = (x == imagePtr->width -1					) ? 0 : imagePtr->pixels[y][x + 1].r;		\
\
	window[2][0] = (x == 0 || y == imagePtr->height - 1			) ? 0 : imagePtr->pixels[y + 1][x - 1].r;	\
	window[2][1] = (y == imagePtr->height - 1				) ? 0 : imagePtr->pixels[y + 1][x].r;		\
	window[2][2] = (x == imagePtr->width -1 || y == imagePtr->height - 1	) ? 0 : imagePtr->pixels[y + 1][x + 1].r;	\
}

 #endif
