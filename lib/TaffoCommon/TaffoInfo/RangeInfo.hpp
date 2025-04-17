#pragma once

#include "../SerializationUtils.hpp"

namespace taffo {

struct Range : public Serializable,
               public Printable {
  double min;
  double max;

  Range()
  : min(0.0), max(0.0) {}
  Range(const Range& other)
  : min(other.min), max(other.max) {}
  Range(double min, double max)
  : min(min), max(max) {}

  bool isConstant() const { return min == max; }
  bool cross(const double val = 0.0) const { return min <= val && max >= val; }

  std::shared_ptr<Range> clone() const { return std::make_shared<Range>(*this); }
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json& j) override;
};

} // namespace taffo
