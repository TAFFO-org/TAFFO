#include "ValueInitInfo.hpp"

using namespace llvm;
using namespace taffo;

std::string ValueInitInfo::toString() const {
  std::stringstream ss;
  ss << "[valueInfo: " << (valueInfo ? valueInfo->toString() : "null")
     << ", rootDistance: " << rootDistance
     << ", backtrackingDepth: " << backtrackingDepth << "]";
  return ss.str();
}
