#pragma once

#include <iostream>
#include <cstring>
#include "bmp.h"


double sqr(double x) {
    return x * x;
}

double distance(const RGB& lhs, const RGB& rhs) {
    return sqrt(sqr(lhs.r - rhs.r) + sqr(lhs.g - rhs.g) + sqr(lhs.b - rhs.b));
}

void compare(const Image &actual, const Image &expected, double threshold = 5) {
    //REQUIRE(actual.nRows == expected.nRows);
    //REQUIRE(actual.nCols == expected.nCols);
    int nPixels = actual.nRows * actual.nCols;
    double mean = 0;
    for (int i = 0; i < nPixels; ++i) {
        mean += distance(actual.pixels[i], expected.pixels[i]);
    }
    mean /= nPixels;
    std::cout << "Distance: " << mean << "\n";
    //REQUIRE(mean <= threshold);
}

double clip_rgb(double val) {
    if (val < 0){
        //printf("Out of range: %f\n", val);
        val= 0;
    }
    if (val > 255){
        //printf("Out of range: %f\n", val);
        val= 255;
    }
    return val;
}


void write_bmp_file( const char *filename, int width, int height, int bytes_per_pixel,  std::vector<RGB> image)
{

    FILE *f;
    long x, y;
    char* bmp = static_cast<char *>(malloc(sizeof(char) * BMP_SIZE(width, height)));


    bmp_init(bmp, width, height);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            float r = clip_rgb(image[y*width + x].r);
            float g = clip_rgb(image[y*width + x].g);
            float b = clip_rgb(image[y*width + x].b);
            bmp_set(bmp, x, y, bmp_encode(r/255, g/255, b/255));
        }
    }

    f = fopen(filename, "wb");
    fwrite(bmp, (sizeof(char) * BMP_SIZE(width, height)), 1, f);
    fclose(f);
}

template <class InputIterator, class OutputIterator>
void fillMatrixInZigzag(InputIterator input, OutputIterator output, int nRows, int nCols) {
    int i = 0;
    int j = 0;
    int di = -1;
    int dj = 1;
    for (int diag = 0; diag < nRows + nCols - 1; ++diag) {
        while (i >= 0 && j >= 0 && i < nRows && j < nCols) {
            *(output + i * nCols + j) = *input;
            ++input;
            i += di;
            j += dj;
        }
        i -= di;
        j -= dj;
        di *= -1;
        dj *= -1;
        if ((i == 0 || i == nRows - 1) && j < nCols - 1) {
            ++j;
        } else {
            ++i;
        }
    }
}


template <typename T>
T clip(T x, T lb, T ub) {
    return std::min(std::max(x, lb), ub);
}
