#include "TransparentType.hpp"

#include "TaffoInfo/TaffoInfo.hpp"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <memory>
#include <sstream>

using namespace llvm;
using namespace taffo;


bool containsPtrType(Type* type){
  if (type->isSingleValueType()) return type->isPointerTy();
  if (type->isArrayTy()){ return containsPtrType(type->getArrayElementType());}
  if ( StructType* structType = dyn_cast<StructType>(type)){
    for (Type* fieldType : structType->elements()){
     if (containsPtrType(fieldType)) return true; 
    }
    return false;
  }
  llvm_unreachable("Type not handled in containsPtrType");
}


std::shared_ptr<TransparentType> TransparentTypeFactory::create(Type *type) 
{ 
  assert(!containsPtrType(type) && "Long life transparent pointer"); 
  return create(type, 0);
}

std::shared_ptr<TransparentType> TransparentTypeFactory::create(const Value *value) {
  assert(!isa<BasicBlock>(value) && "BasicBlock cannot have a transparent type");
  if (auto *function = dyn_cast<Function>(value))
    return create(function->getReturnType(), 0);
  if (auto *global = dyn_cast<GlobalValue>(value))
    return create(global->getValueType(), 0);
  return create(value->getType(), 0);
}

std::shared_ptr<TransparentType> TransparentTypeFactory::create(Type *unwrappedType, unsigned int indirections) {
  if (auto *structType = dyn_cast<StructType>(unwrappedType))
    return std::shared_ptr<TransparentType>(new TransparentStructType(structType, indirections));
  if (auto *arrayType = dyn_cast<ArrayType>(unwrappedType))
    return std::shared_ptr<TransparentType>(new TransparentArrayType(arrayType, indirections));
  if (auto *vectorType = dyn_cast<VectorType>(unwrappedType))
    return std::shared_ptr<TransparentType>(new TransparentArrayType(vectorType, indirections));
  return std::shared_ptr<TransparentType>(new TransparentType(unwrappedType, indirections));
}

std::shared_ptr<TransparentType> TransparentTypeFactory::create(const json &j) {
  std::shared_ptr<TransparentType> type;
  if (j["kind"] == "Struct")
    type = std::shared_ptr<TransparentType>(new TransparentStructType());
  else if (j["kind"] == "Array")
    type = std::shared_ptr<TransparentType>(new TransparentArrayType());
  else
    type = std::shared_ptr<TransparentType>(new TransparentType());
  type->deserialize(j);
  return type;
}

std::shared_ptr<TransparentType> TransparentType::getPointedType() const {
  assert(indirections > 0 && "Not a pointer type or opaque");
  std::shared_ptr<TransparentType> pointedType = clone();
  pointedType->indirections--;
  return pointedType;
}

int TransparentType::compareTransparency(const TransparentType &other) const {
  if (*this == other)
    return 0;

  bool thisOpaque = isOpaquePointer();
  bool otherOpaque = other.isOpaquePointer();
  if (thisOpaque && !otherOpaque)
    return -1;
  if (!thisOpaque && otherOpaque)
    return 1;

  if (indirections < other.indirections)
    return -1;
  if (indirections > other.indirections)
    return 1;

  return 0;
}

Type* TransparentType::toLLVMType() const {
  Type* type = unwrappedType;
  for (unsigned i = 0; i < indirections; ++i)
    type = type->getPointerTo();
  return type;
}

bool TransparentType::operator==(const TransparentType &other) const {
  return getKind() == other.getKind() && unwrappedType == other.unwrappedType && indirections == other.indirections;
}

std::shared_ptr<TransparentType> TransparentType::clone() const {
  return std::shared_ptr<TransparentType>(new TransparentType(*this));
}

std::string TransparentType::toString() const {
  if (!unwrappedType)
    return "InvalidType";
  return taffo::toString(unwrappedType) + std::string(indirections, '*');
}

json TransparentType::serialize() const {
  json j;
  j["repr"] = toString();
  j["kind"] = "Scalar";
  j["unwrappedType"] = taffo::toString(unwrappedType);
  j["indirections"] = indirections;
  return j;
}

void TransparentType::deserialize(const json &j) {
  unwrappedType = TaffoInfo::getInstance().getType(j["unwrappedType"]);
  indirections = j["indirections"];
  assert(unwrappedType != nullptr && "Unwrapped type not found");
}

void TransparentType::incrementIndirections(int increment) {
  if (increment < 0)
    assert(-increment <= static_cast<int>(indirections) && "Indirections underflow");
  else
    assert(indirections <= UINT_MAX - increment && "Indirections overflow");
  indirections += increment;
}

bool TransparentArrayType::isOpaquePointer() const {
  if (TransparentType::isOpaquePointer())
    return true;
  return elementType->isOpaquePointer();
}

int TransparentArrayType::compareTransparency(const TransparentType &other) const {
  if (!isa<TransparentArrayType>(other)) {
    assert(other.isOpaquePointer());
    return 1;
  }
  const auto &otherArray = cast<TransparentArrayType>(other);
  int cmp = TransparentType::compareTransparency(other);
  if (cmp != 0)
    return cmp;
  return  elementType->compareTransparency(*otherArray.elementType);
}

llvm::SmallPtrSet<llvm::Type*, 4> TransparentArrayType::getContainedTypes() const {
  llvm::SmallPtrSet<llvm::Type*, 4> containedTypes = TransparentType::getContainedTypes();
  llvm::SmallPtrSet<llvm::Type*, 4> elementContaineTypes = getArrayElementType()->getContainedTypes();
  containedTypes.insert(elementContaineTypes.begin(), elementContaineTypes.end());   
  return containedTypes;
}

bool TransparentArrayType::operator==(const TransparentType &other) const {
  if (this == &other)
    return true;
  if (getKind() != other.getKind())
    return false;

  const auto &otherArray = cast<TransparentArrayType>(other);
  if (!TransparentType::operator==(other))
    return false;

  if (!elementType && !otherArray.elementType)
    return true;
  if (!elementType || !otherArray.elementType)
    return false;
  return *elementType == *otherArray.elementType;
}

std::shared_ptr<TransparentType> TransparentArrayType::clone() const {
  return std::shared_ptr<TransparentType>(new TransparentArrayType(*this));
}

std::string TransparentArrayType::toString() const {
  if (!unwrappedType || !elementType)
    return "InvalidType";
  std::stringstream ss;
  ss << "[" << *elementType << "]";
  ss << std::string(indirections, '*');
  return ss.str();
}

json TransparentArrayType::serialize() const {
  json j = TransparentType::serialize();
  j["kind"] = "Array";
  j["elementType"] = elementType ? elementType->serialize() : nullptr;
  return j;
}

void TransparentArrayType::deserialize(const json &j) {
  TransparentType::deserialize(j);
  elementType = TransparentTypeFactory::create(j["elementType"]);
}

bool TransparentStructType::isOpaquePointer() const {
  if (TransparentType::isOpaquePointer())
    return true;
  for (const std::shared_ptr<TransparentType> &field : fieldTypes)
    if (!field || field->isOpaquePointer())
      return true;
  return false;
}

bool TransparentStructType::containsFloatingPointType() const  {
  for (const std::shared_ptr<TransparentType> &fieldType : *this)
    if(fieldType->containsFloatingPointType())
      return true;
  return false;
}

int TransparentStructType::compareTransparency(const TransparentType &other) const {
  if (!isa<TransparentStructType>(other)) {
    assert(other.isOpaquePointer());
    return 1;
  }
  const auto &otherStruct = cast<TransparentStructType>(other);
  assert(getNumFieldTypes() == otherStruct.getNumFieldTypes());

  int baseCmp = TransparentType::compareTransparency(other);
  if (baseCmp != 0)
    return baseCmp;

  int overallResult = 0;
  for (unsigned i = 0; i < fieldTypes.size(); i++) {
    int cmp = fieldTypes[i]->compareTransparency(*otherStruct.fieldTypes[i]);
    if (cmp == 0)
      continue;
    if (overallResult == 0)
      overallResult = cmp;
    else if ((overallResult > 0 && cmp < 0) || (overallResult < 0 && cmp > 0))
      return 0; // conflicting fields' comparisons result in equal transparency
  }
  return overallResult;
}

llvm::SmallPtrSet<llvm::Type*, 4> TransparentStructType::getContainedTypes() const {
  llvm::SmallPtrSet<llvm::Type*, 4> containedTypes = TransparentType::getContainedTypes();
  for (auto &field : *this) {
    llvm::SmallPtrSet<llvm::Type*, 4> elementContaineTypes = field->getContainedTypes();
    containedTypes.insert(elementContaineTypes.begin(), elementContaineTypes.end());   
  }
  return containedTypes;
}

bool TransparentStructType::operator==(const TransparentType &other) const {
  if (this == &other)
    return true;
  if (getKind() != other.getKind())
    return false;

  auto &otherStructType = cast<TransparentStructType>(other);
  if (!TransparentType::operator==(other))
    return false;
  if (fieldTypes.size() != otherStructType.fieldTypes.size())
    return false;

  for (unsigned int i = 0; i < fieldTypes.size(); i++) {
    if (!fieldTypes[i] && !otherStructType.fieldTypes[i])
      continue;
    if (!fieldTypes[i] || !otherStructType.fieldTypes[i] || *fieldTypes[i] != *otherStructType.fieldTypes[i])
      return false;
  }
  return true;
}

std::shared_ptr<TransparentType> TransparentStructType::clone() const {
  return std::shared_ptr<TransparentType>(new TransparentStructType(*this));
}

std::string TransparentStructType::toString() const {
  if (!unwrappedType || std::ranges::any_of(fieldTypes,
    [](const std::shared_ptr<TransparentType> &field) -> bool { return field != nullptr; }))
    return "InvalidType";

  std::string typeString = taffo::toString(unwrappedType);
  std::stringstream ss;
  ss << typeString.substr(0, typeString.find('{') + 1) << " ";

  bool first = true;
  for (const auto &fieldType : fieldTypes) {
    if (!first)
      ss << ", ";
    else
      first = false;
    ss << *fieldType;
  }

  ss << " }" << std::string(indirections, '*');
  return ss.str();
}

json TransparentStructType::serialize() const {
  json j = TransparentType::serialize();
  j["kind"] = "Struct";
  j["fieldTypes"] = json::array();
  for (const auto &field : fieldTypes)
    j["fieldTypes"].push_back(field ? field->serialize() : nullptr);
  return j;
}

void TransparentStructType::deserialize(const json &j) {
  TransparentType::deserialize(j);
  fieldTypes.clear();
  for (const auto &f : j["fieldTypes"])
    fieldTypes.push_back(TransparentTypeFactory::create(f));
}
