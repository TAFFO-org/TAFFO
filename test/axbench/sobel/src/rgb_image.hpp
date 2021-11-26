/*
 * rgb_image.hpp
 *
 * Created on: Sep 9, 2013
 * 			Author: Amir Yazdanbakhsh <a.yazdanbakhsh@gatech.edu>
 */

#ifndef __RGB_IMAGE_HPP__
#define __RGB_IMAGE_HPP__

#include <vector>
#include <fstream>
#include <stdlib.h>
#include <iostream>

#define ANNOTATION_IMAGE "struct[void,void,struct[scalar(),scalar(),scalar()],void]"
#define ANNOTATION_IMAGE_RANGE "struct[void,void,struct[scalar(range(0,255)),scalar(range(0,255)),scalar(range(0,255))],void]"

#define DEBUG 0

class Pixel {
public:
	Pixel () {}
	Pixel (float r, float g, float b)
	{
		this->r = r ;
		this->g = g ;
		this->b = b ;
	}
	float r ;
	float g ;
	float b ;
} ;


class Image {
public:
	int 			width ;
	int 			height ;
	Pixel**			pixels ;
	//std::vector 	<std::vector<boost::shared_ptr<Pixel> > > pixels ;
	std::string 	meta ;

	// Constructor
	Image()
	{
		this->width  = 0 ;
		this->height = 0 ;
	}

	int loadRgbImage (std::string filename) ;
	int saveRgbImage (std::string outFilename, float scale) ;
	void makeGrayscale() ;
	void printPixel(int x, int y) ;

} ;

#endif

