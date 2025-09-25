#pragma once

#include "TaffoInfo/TaffoInfo.hpp"
#include "TaffoInfo/ValueInfo.hpp"

namespace taffo {

class ValueInitInfoFactory;
class TaffoInitInfo;

class ValueInitInfo : tda::Printable {
public:
  friend class ValueInitInfoFactory;

  ValueInitInfo() = delete;

  void setRootDistance(unsigned distance) { rootDistance = distance; }
  unsigned getRootDistance() const { return rootDistance; }
  unsigned getUserRootDistance() const { return std::max(rootDistance, rootDistance + 1); }

  std::string toString() const override;

private:
  unsigned rootDistance;

  ValueInitInfo(unsigned rootDistance)
  : rootDistance(rootDistance) {}
};

class ValueInitInfoFactory {
private:
  friend class TaffoInitInfo;

  static ValueInitInfo createValueInitInfo(unsigned rootDistance) { return ValueInitInfo(rootDistance); }
};

} // namespace taffo
