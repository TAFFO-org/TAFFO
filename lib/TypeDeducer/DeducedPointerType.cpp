#include "DeducedPointerType.hpp"

#include "TaffoInfo/TaffoInfo.hpp"

using namespace taffo;

std::string DeducedPointerType::toString() const {
  if (isAmbiguous())
    return "AmbiguousType";
  return taffo::toString(unwrappedType) + std::string(indirections, '*');
}


json DeducedPointerType::serialize() const {
  json j;
  j["repr"] = toString();
  j["unwrappedType"] = taffo::toString(unwrappedType);
  j["indirections"] = indirections;
  return j;
}

void DeducedPointerType::deserialize(const json &j) {
  unwrappedType = TaffoInfo::getInstance().getType(j["unwrappedType"]);
  indirections = j["indirections"];
}
