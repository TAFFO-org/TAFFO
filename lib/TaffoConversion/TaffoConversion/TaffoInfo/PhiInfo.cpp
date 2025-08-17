#include "PhiInfo.hpp"

using namespace llvm;
using namespace taffo;

std::string PhiInfo::toString() const {
  std::string ss;
  raw_string_ostream os(ss);
  os << "{ placeholderNoConv: ";
  os << *oldPhi;
  os << "placeholderConv ";
  os << *newPhi;
  os << " }";
  return ss;
}
