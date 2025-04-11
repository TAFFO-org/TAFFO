#include "ValueInitInfo.hpp"

using namespace llvm;
using namespace taffo;

std::string ValueInitInfo::toString() const {
  std::stringstream ss;
  ss << "(backtrackingDepthLeft: " << backtrackingDepthLeft
     << ", rootDistance: " << rootDistance
     << ", valueInfo: " << (valueInfo ? valueInfo->toString() : "null") << ")";
  return ss.str();
}
