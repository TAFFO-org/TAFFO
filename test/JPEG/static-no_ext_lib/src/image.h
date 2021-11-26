#pragma once

#include <vector>
#include <string>


struct RGB {
    int r, g, b;
};


struct Image {
    int nRows;
    int nCols;
    std::vector<RGB> pixels;
};
