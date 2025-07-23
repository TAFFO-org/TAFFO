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
  j["min"] = serializeDouble(min);
  j["max"] = serializeDouble(max);
  return j;
}

void Range::deserialize(const json& j) {
  min = deserializeDouble(j["min"]);
  max = deserializeDouble(j["max"]);
}
