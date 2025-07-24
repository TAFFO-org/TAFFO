#include "SerializationUtils.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "Utils/PrintUtils.hpp"

using namespace llvm;
using namespace tda;
using namespace taffo;

static constexpr std::string_view infStr = "inf";
static constexpr std::string_view ninfStr = "-inf";
static constexpr std::string_view nanStr = "nan";

json taffo::serializeDouble(double value) {
  if (std::isfinite(value))
    return value;
  if (std::isnan(value))
    return json(nanStr);
  if (value > 0)
    return json(infStr);
  return json(ninfStr);
}

double taffo::deserializeDouble(const json& j) {
  if (j.is_number())
    return j.get<double>();
  auto s = j.get<std::string_view>();
  if (s == infStr)
    return std::numeric_limits<double>::infinity();
  if (s == ninfStr)
    return -std::numeric_limits<double>::infinity();
  if (s == nanStr)
    return std::numeric_limits<double>::quiet_NaN();
  llvm_unreachable("Unknown value");
}

json serializeCommon(const TransparentType& type) {
  json j;
  j["kind"] = "Scalar";
  j["repr"] = type.toString();
  j["unwrappedType"] = toString(type.unwrappedType);
  j["indirections"] = type.indirections;
  return j;
}

json taffo::serialize(const TransparentType& type) {
  if (auto* arrayType = dyn_cast<const TransparentArrayType>(&type))
    return serialize(*arrayType);
  if (auto* structType = dyn_cast<const TransparentStructType>(&type))
    return serialize(*structType);
  return serializeCommon(*cast<const TransparentType>(&type));
}

json taffo::serialize(const TransparentArrayType& arrayType) {
  json j = serializeCommon(arrayType);
  j["kind"] = "Array";
  j["elementType"] = arrayType.elementType ? serialize(*arrayType.elementType) : nullptr;
  return j;
}

json taffo::serialize(const TransparentStructType& structType) {
  json j = serializeCommon(structType);
  j["kind"] = "Struct";
  j["fieldTypes"] = json::array();
  for (auto& f : structType.fieldTypes)
    j["fieldTypes"].push_back(f ? serialize(*f) : nullptr);
  j["paddingFields"] = structType.paddingFields;
  return j;
}

void deserializeCommon(const json& j, TransparentType& type) {
  type.unwrappedType = TaffoInfo::getInstance().getType(j["unwrappedType"]);
  type.indirections = j["indirections"];
}

std::shared_ptr<TransparentType> taffo::deserialize(const json& j) {
  std::shared_ptr<TransparentType> type;
  const std::string kind = j["kind"];
  if (kind == "Struct")
    type = std::make_shared<TransparentStructType>();
  else if (kind == "Array")
    type = std::make_shared<TransparentArrayType>();
  else
    type = std::make_shared<TransparentType>();

  deserializeCommon(j, *type);

  if (kind == "Struct") {
    auto structType = std::static_ptr_cast<TransparentStructType>(type);
    for (auto& field_j : j["fieldTypes"])
      structType->fieldTypes.push_back(deserialize(field_j));
    if (j.contains("paddingFields"))
      for (unsigned padding : j["paddingFields"])
        structType->paddingFields.push_back(padding);
  }
  else if (kind == "Array") {
    auto arrayType = std::static_ptr_cast<TransparentArrayType>(type);
    arrayType->elementType = deserialize(j["elementType"]);
  }

  return type;
}
