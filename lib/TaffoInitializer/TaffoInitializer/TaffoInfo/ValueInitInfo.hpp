#pragma once

#include "TaffoInfo/TaffoInfo.hpp"
#include "TaffoInfo/ValueInfo.hpp"

namespace taffo {

class ValueInitInfoFactory;
class TaffoInitInfo;

class ValueInitInfo : Printable {
public:
  friend class ValueInitInfoFactory;

  ValueInitInfo() = delete;

  void setRootDistance(unsigned distance) { rootDistance = distance; }
  unsigned getRootDistance() const { return rootDistance; }
  unsigned getUserRootDistance() const { return std::max(rootDistance, rootDistance + 1); }
  void setBacktrackingDepth(unsigned depth) { backtrackingDepth = depth; }
  void decreaseBacktrackingDepth() { backtrackingDepth = std::min(backtrackingDepth, backtrackingDepth - 1); }
  unsigned getBacktrackingDepth() const { return backtrackingDepth; }
  unsigned getUserBacktrackingDepth() const { return std::min(backtrackingDepth, backtrackingDepth - 1); }

  std::string toString() const override;

private:
  unsigned rootDistance;
  unsigned backtrackingDepth;

  ValueInitInfo(unsigned rootDistance, unsigned backtrackingDepth)
  : rootDistance(rootDistance), backtrackingDepth(backtrackingDepth) {}
};

class ValueInitInfoFactory {
private:
  friend class TaffoInitInfo;

  static ValueInitInfo createValueInitInfo(unsigned rootDistance, unsigned backtrackingDepth) {
    return ValueInitInfo(rootDistance, backtrackingDepth);
  }
};

} // namespace taffo
