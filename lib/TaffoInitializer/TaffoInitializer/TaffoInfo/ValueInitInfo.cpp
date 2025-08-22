#include "ValueInitInfo.hpp"

using namespace llvm;
using namespace taffo;

std::string ValueInitInfo::toString() const {
  std::stringstream ss;
  ss << "[rootDistance: " << rootDistance << ", backtrackingDepth: " << backtrackingDepth << "]";
  return ss.str();
}
