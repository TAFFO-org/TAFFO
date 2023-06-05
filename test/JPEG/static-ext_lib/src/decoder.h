#pragma once

#include "image.h"


Image decode(const std::string& filename);
Image decode(std::istream& stream);
