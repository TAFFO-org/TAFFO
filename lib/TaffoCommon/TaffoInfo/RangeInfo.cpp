#include "RangeInfo.hpp"

using namespace llvm;
using namespace taffo;

std::string Range::toString() const {
  std::stringstream ss;
  ss << "[" << Min << ", " << Max << "]";
  return ss.str();
}

json Range::serialize() const {
  json j;
  j["Min"] = Min;
  j["Max"] = Max;
  return j;
}

void Range::deserialize(const json &j) {
  Min = j["Min"].get<double>();
  Max = j["Max"].get<double>();
}
