#include "SerializationUtils.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "Utils/PrintUtils.hpp"

#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>

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
  j["unwrappedType"] = nullptr;
  if (const Type* llvmType = type.getLLVMType())
    j["unwrappedType"] = toString(llvmType);
  j["isUnion"] = type.isUnion();
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

const TransparentType* taffo::deserialize(const json& j, llvm::LLVMContext* llvmContext) {
  const TransparentType* type = nullptr;

  const llvm::Type* llvmType = nullptr;
  if (!j["unwrappedType"].is_null())
    llvmType = TaffoInfo::getInstance().getType(j["unwrappedType"]);

  const bool isUnion = j["isUnion"];

  const std::string kind = j["kind"];
  if (kind == "Scalar") {
    type = TransparentType::get(llvmContext, llvmType, isUnion);
  }
  else if (kind == "Pointer") {
    const TransparentType* pointedType = nullptr;
    if (j.contains("pointedType") && !j["pointedType"].is_null())
      pointedType = deserialize(j["pointedType"], llvmContext);

    type = TransparentPointerType::get(llvmContext, pointedType);
  }
  else if (kind == "Array") {
    const TransparentType* elementType = nullptr;
    if (j.contains("elementType") && !j["elementType"].is_null())
      elementType = deserialize(j["elementType"], llvmContext);

    type = TransparentArrayType::get(llvmContext, elementType, llvmType);
  }
  else if (kind == "Struct") {
    llvm::SmallVector<const TransparentType*, 8> fieldTypes;
    for (const auto& field_j : j["fieldTypes"]) {
      const TransparentType* fieldType = deserialize(field_j, llvmContext);
      fieldTypes.push_back(fieldType);
    }

    llvm::SmallSet<unsigned, 8> paddingFields;
    if (j.contains("paddingFields"))
      for (unsigned paddingField : j["paddingFields"])
        paddingFields.insert(paddingField);

    type = TransparentStructType::get(llvmContext, fieldTypes, llvmType, {}, {}, paddingFields);
  }

  return type;
}
