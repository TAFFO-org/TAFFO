#include "RangeInfo.hpp"

using namespace llvm;
using namespace taffo;

std::string Range::toString() const {
  std::stringstream ss;
  ss << "[" << min << ", " << max << "]";
  return ss.str();
}

json Range::serialize() const {
  json j;
  j["Min"] = min;
  j["Max"] = max;
  return j;
}

void Range::deserialize(const json& j) {
  min = j["Min"].get<double>();
  max = j["Max"].get<double>();
}
