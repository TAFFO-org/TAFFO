#pragma once

#include "TaffoInfo/ValueInfo.hpp"

namespace taffo {

class ValueInitInfoFactory;
class TaffoInitInfo;

class ValueInitInfo : Printable {
public:
  friend class ValueInitInfoFactory;

  ValueInfo *getValueInfo() { return valueInfo; }
  void setRootDistance(unsigned int distance) { rootDistance = distance; }
  unsigned int getRootDistance() const { return rootDistance; }
  unsigned int getUserRootDistance() const { return std::max(rootDistance, rootDistance + 1); }
  void setBacktrackingDepth(unsigned int depth) { backtrackingDepth = depth; }
  void decreaseBacktrackingDepth() { backtrackingDepth = std::min(backtrackingDepth, backtrackingDepth - 1); }
  unsigned int getBacktrackingDepth() const { return backtrackingDepth; }
  unsigned int getUserBacktrackingDepth() const { return std::min(backtrackingDepth, backtrackingDepth - 1); }

  std::string toString() const override;

private:
  ValueInfo *valueInfo;
  unsigned int rootDistance;
  unsigned int backtrackingDepth;

  ValueInitInfo() = delete;

  ValueInitInfo(ValueInfo *valueInfo, unsigned int rootDistance, unsigned int backtrackingDepth)
  : valueInfo(valueInfo), rootDistance(rootDistance), backtrackingDepth(backtrackingDepth) {}
};

class ValueInitInfoFactory {
private:
  friend class TaffoInitInfo;

  static ValueInitInfo createValueInitInfo(ValueInfo *valueInfo, unsigned int rootDistance, unsigned int backtrackingDepth) {
    return ValueInitInfo(valueInfo, rootDistance, backtrackingDepth);
  }
};

} // namespace taffo
