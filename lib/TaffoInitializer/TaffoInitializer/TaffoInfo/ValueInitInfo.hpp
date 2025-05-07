#pragma once

#include "TaffoInfo/ValueInfo.hpp"

namespace taffo {

class ValueInitInfoFactory;
class TaffoInitInfo;

class ValueInitInfo : Printable {
public:
  friend class ValueInitInfoFactory;

  ValueInfo* getValueInfo() { return valueInfo; }
  void setRootDistance(unsigned distance) { rootDistance = distance; }
  unsigned getRootDistance() const { return rootDistance; }
  unsigned getUserRootDistance() const { return std::max(rootDistance, rootDistance + 1); }
  void setBacktrackingDepth(unsigned depth) { backtrackingDepth = depth; }
  void decreaseBacktrackingDepth() { backtrackingDepth = std::min(backtrackingDepth, backtrackingDepth - 1); }
  unsigned getBacktrackingDepth() const { return backtrackingDepth; }
  unsigned getUserBacktrackingDepth() const { return std::min(backtrackingDepth, backtrackingDepth - 1); }

  std::string toString() const override;

private:
  ValueInfo* valueInfo;
  unsigned rootDistance;
  unsigned backtrackingDepth;

  ValueInitInfo() = delete;

  ValueInitInfo(ValueInfo* valueInfo, unsigned rootDistance, unsigned backtrackingDepth)
  : valueInfo(valueInfo), rootDistance(rootDistance), backtrackingDepth(backtrackingDepth) {}
};

class ValueInitInfoFactory {
private:
  friend class TaffoInitInfo;

  static ValueInitInfo createValueInitInfo(ValueInfo* valueInfo, unsigned rootDistance, unsigned backtrackingDepth) {
    return ValueInitInfo(valueInfo, rootDistance, backtrackingDepth);
  }
};

} // namespace taffo
