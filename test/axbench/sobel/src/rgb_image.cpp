/*
 * rgb_image.cpp
 * 
 * Created on: Sep 9, 2013
 * 			Author: Amir Yazdanbakhsh <a.yazdanbakhsh@gatech.edu>
 */

#include "rgb_image.hpp"
#include <cstdlib>

void Image::printPixel(int x, int y)
{
	std::cout << "# Red: 	" << this->pixels[x][y].r << std::endl;
	std::cout << "# Green: 	" << this->pixels[x][y].g << std::endl;
	std::cout << "# Blue: 	" << this->pixels[x][y].b << std::endl; 
}


const char *readInt(const char *line, int *out)
{
  while (*line != '\0') {
    if ('0' <= *line && *line <= '9') {
      char *end;
      *out = strtol(line, &end, 0);
      return end;
    } else {
      line++;
    }
  }
  return nullptr;
}


void tokenize(std::vector<int>& out, std::string& line)
{
  const char *str = line.c_str();
  int v;
  str = readInt(str, &v);
  while (str) {
    out.push_back(v);
    str = readInt(str, &v);
  }
}


int Image::loadRgbImage(std::string filename)
{
	std::ifstream imageFile ;

	if(DEBUG)
		std::cout << "# Loading " << filename << " ..." << std::endl ;

	imageFile.open(filename.c_str()) ;
	if(! imageFile.is_open())
	{
		std::cerr << "# Error openning the file!" << std::endl ;
		return 0 ;
	}

	// Read first line and split based on the , and any spaces before or after
	std::string line ;
	std::getline(imageFile, line) ;
	std::vector<int> imageInfo ;
	tokenize(imageInfo, line);
	this->width  = imageInfo[0];
	this->height = imageInfo[1];

	if(DEBUG)
	{
		std::cout << "# Width:  " << this->width ;
		std::cout << "# Height: " << this->height  << std::endl ;
	} 

	//this->_pixels = (void*)malloc(this->height * (this->width * 3) * sizeof(float));
	this->pixels = new Pixel*[this->height];

	// We assume there is a newline after each row
	for (int h = 0 ; h < this->height ; h++)
	{
		std::getline(imageFile, line) ;
		std::vector<int> currRowString;
		tokenize(currRowString, line);
		this->pixels[h] = new Pixel[this->width];

		for(int w = 0 ; w < this->width ; w++)
		{
			int index = w * 3 ;
			float __attribute((annotate("scalar(range(0,255))"))) r = currRowString[index++];
			float __attribute((annotate("scalar(range(0,255))"))) g = currRowString[index++];
			float __attribute((annotate("scalar(range(0,255))"))) b = currRowString[index++];
			// Add pixel to the current row
			this->pixels[h][w].r = r ;
			this->pixels[h][w].g = g ;
			this->pixels[h][w].b = b ;
		}
	}

	std::getline(imageFile, line) ;
	this->meta = line ;
	return 1 ;
}

int Image::saveRgbImage(std::string outFilename,
			float __attribute((annotate("errtarget('out') scalar()"))) scale) // range(362,362)
{
	if(DEBUG)
	{
		std::cout << "# Saving into " << outFilename << " ..." << std::endl ;
	}

	std::ofstream outFile ;
	outFile.open(outFilename.c_str()) ;

	outFile << this->width << "," << this->height << std::endl ;

	for(int h = 0 ; h < this->height ; h++)
	{
		for(int w = 0 ; w < (this->width - 1); w++)
		{
			// Write Red
			int red   = (int)(this->pixels[h][w].r * scale) ;
			int green = (int)(this->pixels[h][w].g * scale) ;
			int blue  = (int)(this->pixels[h][w].b * scale) ;

			//if ( red > 255 )
		//		red = 255 ;
		//	if ( green > 255 )
		//		green = 255 ;
		///	if ( blue > 255 )
			//	blue = 255 ;
			outFile << red << "," ;
			// Write Green
			outFile << green << "," ;
			// Write Blue
			outFile << blue << "," ;
		}

		int red   = (int)(this->pixels[h][this->width - 1].r * scale) ;
		int green = (int)(this->pixels[h][this->width - 1].g * scale);
		int blue  = (int)(this->pixels[h][this->width - 1].b * scale) ;


		// Write Red
		outFile << red  << "," ;
		// Write Green
		outFile << green << "," ;
		// Write Blue
		outFile << blue << std::endl ;
	}

	// Print the meta information
	outFile << this->meta ;
	outFile.close() ;
	return 1 ;
}

void Image::makeGrayscale()
{

	float __attribute((annotate("scalar()"))) luminance ;

	float __attribute((annotate("scalar()"))) rC = 0.30 / 256.0 ;
	float __attribute((annotate("scalar()"))) gC = 0.59 / 256.0 ;
	float __attribute((annotate("scalar()"))) bC = 0.11 / 256.0 ;

	for(int h = 0 ; h < this->height ; h++)
	{
		for(int w = 0 ; w < this->width ; w++)
		{
			luminance = ( rC * this->pixels[h][w].r ) +
			  ( gC * this->pixels[h][w].g ) +
			  ( bC * this->pixels[h][w].b ) ;

			this->pixels[h][w].r = luminance ;
			this->pixels[h][w].g = luminance ;
			this->pixels[h][w].b = luminance ;
		}
	}
}
