#ifndef TAFFO_VALUE_INIT_INFO_HPP
#define TAFFO_VALUE_INIT_INFO_HPP

#include "TaffoInfo/ValueInfo.hpp"

namespace taffo {

class TaffoInitializerPass;

class ValueInitInfo : Printable {
private:
  friend class TaffoInitializerPass;

  unsigned int rootDistance = UINT_MAX;
  unsigned int backtrackingDepthLeft = 0;
  ValueInfo *valueInfo;

  ValueInitInfo(ValueInfo *valueInfo)
  : valueInfo(valueInfo) {}

  void setRootDistance(unsigned int distance) { rootDistance = distance; }
  void setBacktrackingDepthLeft(unsigned int depth) { backtrackingDepthLeft = depth; }

  std::string toString() const override;
};

}

#endif // TAFFO_VALUE_INIT_INFO_HPP
