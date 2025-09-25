#include "ValueInitInfo.hpp"

using namespace llvm;
using namespace taffo;

std::string ValueInitInfo::toString() const {
  std::stringstream ss;
  ss << "[rootDistance: ";
  if (rootDistance == UINT_MAX)
    ss << "inf";
  else
    ss << rootDistance;
  ss << "]";
  return ss.str();
}
