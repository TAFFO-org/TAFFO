#include "SerializationUtils.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "Utils/PrintUtils.hpp"

using namespace llvm;
using namespace tda;
using namespace taffo;

static constexpr std::string_view infStr = "inf";
static constexpr std::string_view nInfStr = "-inf";
static constexpr std::string_view nanStr = "nan";

json taffo::serializeDouble(double value) {
  if (std::isfinite(value))
    return value;
  if (std::isnan(value))
    return json(nanStr);
  if (value > 0)
    return json(infStr);
  return json(nInfStr);
}

double taffo::deserializeDouble(const json& j) {
  if (j.is_number())
    return j.get<double>();
  auto s = j.get<std::string_view>();
  if (s == infStr)
    return std::numeric_limits<double>::infinity();
  if (s == nInfStr)
    return -std::numeric_limits<double>::infinity();
  if (s == nanStr)
    return std::numeric_limits<double>::quiet_NaN();
  llvm_unreachable("Unknown value");
}

json serializeCommon(const TransparentType& type) {
  json j;
  j["kind"] = "Scalar";
  j["repr"] = type.toString();
  j["unwrappedType"] = toString(type.getLLVMType());
  return j;
}

json taffo::serialize(const TransparentType& type) {
  if (auto* ptrTy = dyn_cast<const TransparentPointerType>(&type))
    return serialize(*ptrTy);
  if (auto* arrayType = dyn_cast<const TransparentArrayType>(&type))
    return serialize(*arrayType);
  if (auto* structType = dyn_cast<const TransparentStructType>(&type))
    return serialize(*structType);
  return serializeCommon(*cast<const TransparentType>(&type));
}

json taffo::serialize(const TransparentPointerType& ptrType) {
  json j = serializeCommon(ptrType);
  j["kind"] = "Pointer";
  const TransparentType* pointedType = ptrType.getPointedType();
  j["pointedType"] = pointedType ? serialize(*pointedType) : nullptr;
  return j;
}

json taffo::serialize(const TransparentArrayType& arrayType) {
  json j = serializeCommon(arrayType);
  j["kind"] = "Array";
  j["elementType"] = arrayType.getElementType() ? serialize(*arrayType.getElementType()) : nullptr;
  return j;
}

json taffo::serialize(const TransparentStructType& structType) {
  json j = serializeCommon(structType);
  j["kind"] = "Struct";
  j["fieldTypes"] = json::array();
  for (const TransparentType* field : structType.getFieldTypes())
    j["fieldTypes"].push_back(field ? serialize(*field) : nullptr);
  j["paddingFields"] = structType.getPaddingFields();
  return j;
}

void deserializeCommon(const json& j, TransparentType& type) {
  type.setLLVMType(TaffoInfo::getInstance().getType(j["unwrappedType"]));
}

std::unique_ptr<TransparentType> taffo::deserialize(const json& j) {
  std::unique_ptr<TransparentType> type;
  const std::string kind = j["kind"];
  if (kind == "Struct")
    type = std::make_unique<TransparentStructType>();
  else if (kind == "Array")
    type = std::make_unique<TransparentArrayType>();
  else if (kind == "Pointer")
    type = std::make_unique<TransparentPointerType>();
  else
    type = std::make_unique<TransparentType>();

  deserializeCommon(j, *type);

  if (kind == "Struct") {
    auto* structType = cast<TransparentStructType>(type.get());
    for (auto& field_j : j["fieldTypes"])
      structType->addFieldType(deserialize(field_j));
    if (j.contains("paddingFields"))
      for (unsigned paddingField : j["paddingFields"])
        structType->addFieldPadding(paddingField);
  }
  else if (kind == "Array") {
    auto* arrayType = cast<TransparentArrayType>(type.get());
    arrayType->setElementType(deserialize(j["elementType"]));
  }
  else if (kind == "Pointer") {
    auto* ptrType = cast<TransparentPointerType>(type.get());
    if (j.contains("pointedType") && !j["pointedType"].is_null())
      ptrType->setPointedType(deserialize(j["pointedType"]));
  }

  return type;
}
