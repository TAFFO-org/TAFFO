#pragma once

#include "../SerializationUtils.hpp"

namespace taffo {

struct Range : public Serializable, public Printable {
  double Min;
  double Max;

  Range() : Min(0.0), Max(0.0) {}
  Range(const Range &other) : Min(other.Min), Max(other.Max) {}
  Range(double Min, double Max) : Min(Min), Max(Max) {}

  bool isConstant() const { return Min == Max; }
  bool cross(const double val = 0.0) const {
    return Min <= val && Max >= val;
  }

  std::shared_ptr<Range> clone() const {
    return std::make_shared<Range>(*this);
  }
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json &j) override;
};

}
