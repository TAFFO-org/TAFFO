#ifndef TAFFO_VALUE_INIT_INFO_HPP
#define TAFFO_VALUE_INIT_INFO_HPP

#include "Initializer/TaffoInitializer/TaffoInfo/TaffoInitInfo.hpp"
#include "TaffoInfo/ValueInfo.hpp"

namespace taffo {

class TaffoInitializerPass;
class TaffoInitInfo;

class ValueInitInfo : Printable {
private:
  friend class TaffoInitializerPass;
  friend class TaffoInitInfo;

  ValueInfo *valueInfo;
  unsigned int rootDistance = UINT_MAX;
  unsigned int backtrackingDepthLeft = 0;

  ValueInitInfo(ValueInfo *valueInfo)
  : valueInfo(valueInfo) {}


  ValueInitInfo(ValueInfo *valueInfo, unsigned int rootDistance, unsigned int backtrackingDepthLeft )
  : valueInfo(valueInfo), rootDistance(rootDistance), backtrackingDepthLeft(backtrackingDepthLeft) {}

  void setRootDistance(unsigned int distance) { rootDistance = distance; }
  void setBacktrackingDepthLeft(unsigned int depth) { backtrackingDepthLeft = depth; }

  std::string toString() const override;
};

}

#endif // TAFFO_VALUE_INIT_INFO_HPP
