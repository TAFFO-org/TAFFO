#include "ConversionType.hpp"
#include "TransparentType.hpp"
#include "Utils/PtrCasts.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/Casting.h>

#include <memory>
#include <sstream>

#define DEBUG_TYPE "taffo-conversion"

using namespace llvm;
using namespace tda;
using namespace taffo;

std::unique_ptr<ConversionType> ConversionTypeFactory::create(const TransparentType& type) {
  const TransparentType* currentType = &type;
  if (currentType->isStructTT()) {
    auto* structType = cast<TransparentStructType>(currentType);
    SmallVector<std::unique_ptr<ConversionType>, 4> fields;
    fields.reserve(structType->getNumFieldTypes());
    for (unsigned i = 0; i < structType->getNumFieldTypes(); i++)
      fields.push_back(create(*structType->getFieldType(i)));
    return std::make_unique<ConversionStructType>(*cast<TransparentStructType>(currentType), fields);
  }
  return std::make_unique<ConversionScalarType>(*currentType);
}

TransparentType* ConversionType::toTransparentType(bool* hasFloats) const {
  if (!recomputedTransparentType) {
    recomputedTransparentType = true;
    this->hasFloats = toTransparentTypeHelper(*transparentType);
  }
  if (hasFloats)
    *hasFloats = this->hasFloats;
  return transparentType.get();
}

std::unique_ptr<ConversionType> ConversionType::getGepConvType(const ArrayRef<unsigned> gepIndices) const {
  const TransparentType* resolvedType = transparentType.get();
  const ConversionType* resolvedConvType = this;
  unsigned indirections = resolvedType->getIndirections();
  for (unsigned index : gepIndices)
    if (indirections > 0)
      indirections--;
    else if (resolvedType->isStructTT()) {
      resolvedType = cast<TransparentStructType>(resolvedType)->getFieldType(index);
      indirections = resolvedType->getIndirections();
      resolvedConvType = cast<ConversionStructType>(resolvedConvType)->getFieldType(index);
    }
    else if (resolvedType->isArrayTT()) {
      resolvedType = cast<TransparentArrayType>(resolvedType)->getArrayElementType();
      indirections = resolvedType->getIndirections();
    }
    else
      llvm_unreachable("Unsupported type in gep");

  if (!resolvedConvType)
    return nullptr;

  indirections++;
  if (indirections != resolvedType->getIndirections()) {
    std::unique_ptr<TransparentType> resolvedTypeCopy = resolvedType->clone();
    resolvedTypeCopy->setIndirections(indirections);
    return resolvedConvType->clone(*resolvedTypeCopy);
  }
  return resolvedConvType->clone(*resolvedType);
}

std::unique_ptr<ConversionType> ConversionType::getGepConvType(const iterator_range<const Use*> gepIndices) const {
  SmallVector<unsigned, 4> indicesVector;
  for (Value* value : gepIndices) {
    auto constantIndex = dyn_cast<ConstantInt>(value);
    // The constant value of the index is only used to navigate struct types.
    // In other cases indicesVector is only used to count pointer indirections,
    // so only its cardinality matters and not the values themselves
    indicesVector.push_back(constantIndex ? constantIndex->getZExtValue() : 0);
  }
  return getGepConvType(indicesVector);
}

ConversionType& ConversionType::operator=(const ConversionType& other) {
  if (this == &other)
    return *this;
  transparentType = other.transparentType ? other.transparentType->clone() : nullptr;
  recomputedTransparentType = other.recomputedTransparentType;
  hasFloats = other.hasFloats;
  return *this;
}

ConversionScalarType::ConversionScalarType(const TransparentType& type, bool isSigned)
: ConversionType(type), sign(isSigned) {
  assert(type.isScalarTT() || type.isArrayTT());
  const TransparentType* curr = &type;
  while (curr->isArrayTT())
    curr = cast<TransparentArrayType>(curr)->getArrayElementType();
  Type* unwrappedType = curr->getUnwrappedLLVMType();
  if (unwrappedType->isFloatingPointTy()) {
    bits = 0;
    fractionalBits = 0;
    if (unwrappedType->getTypeID() == Type::HalfTyID)
      floatStandard = Float_half;
    else if (unwrappedType->getTypeID() == Type::DoubleTyID)
      floatStandard = Float_double;
    else if (unwrappedType->getTypeID() == Type::FloatTyID)
      floatStandard = Float_float;
    else if (unwrappedType->getTypeID() == Type::FP128TyID)
      floatStandard = Float_fp128;
    else if (unwrappedType->getTypeID() == Type::PPC_FP128TyID)
      floatStandard = Float_ppc_fp128;
    else if (unwrappedType->getTypeID() == Type::X86_FP80TyID)
      floatStandard = Float_x86_fp80;
    else if (unwrappedType->getTypeID() == Type::BFloatTyID)
      floatStandard = Float_bfloat;
    else
      floatStandard = NotFloat;
  }
  else if (unwrappedType->isIntegerTy()) {
    bits = unwrappedType->getIntegerBitWidth();
    fractionalBits = 0;
    floatStandard = NotFloat;
  }
  else {
    sign = false;
    bits = 0;
    fractionalBits = 0;
    floatStandard = NotFloat;
  }
}

ConversionScalarType::ConversionScalarType(const TransparentType& type, NumericTypeInfo* numericType)
: ConversionType(type) {
  if (numericType) {
    if (auto* fixedPointInfo = dyn_cast<FixedPointInfo>(numericType)) {
      bits = fixedPointInfo->getBits();
      fractionalBits = fixedPointInfo->getFractionalBits();
      sign = fixedPointInfo->isSigned();
      floatStandard = NotFloat;
    }
    else if (auto* floatingPointInfo = dyn_cast<FloatingPointInfo>(numericType)) {
      bits = 0;
      fractionalBits = 0;
      sign = true;
      floatStandard = static_cast<FloatStandard>(floatingPointInfo->getStandard());
    }
    else {
      sign = false;
      bits = 0;
      fractionalBits = 0;
      floatStandard = NotFloat;
    }
  }
  else {
    sign = false;
    bits = 0;
    fractionalBits = 0;
    floatStandard = NotFloat;
  }
}

Type* ConversionScalarType::toScalarLLVMType(LLVMContext& context) const {
  if (floatStandard == NotFloat)
    return Type::getIntNTy(context, bits);
  switch (floatStandard) {
  case Float_half:      // 16-bit floating-point value
    return Type::getHalfTy(context);
  case Float_float:     // 32-bit floating-point value
    return Type::getFloatTy(context);
  case Float_double:    // 64-bit floating-point value
    return Type::getDoubleTy(context);
  case Float_fp128:     // 128-bit floating-point value (112-bit mantissa)
    return Type::getFP128Ty(context);
  case Float_x86_fp80:  // 80-bit floating-point value (X87)
    return Type::getX86_FP80Ty(context);
  case Float_ppc_fp128: // 128-bit floating-point value (two 64-bits)
    return Type::getPPC_FP128Ty(context);
  case Float_bfloat:    // 128-bit floating-point value (two 64-bits)
    return Type::getBFloatTy(context);
  default: llvm_unreachable("Unhandled floating point type");
  }
}

bool ConversionScalarType::toTransparentTypeHelper(TransparentType& newType) const {
  bool hasFloats = false;
  if (newType.isArrayTT()) {
    // Array case
    auto& arrType = cast<TransparentArrayType>(newType);
    hasFloats = toTransparentTypeHelper(*arrType.getArrayElementType());
    newType.setUnwrappedLLVMType(ArrayType::get(arrType.getArrayElementType()->getUnwrappedLLVMType(),
                                                newType.getUnwrappedLLVMType()->getArrayNumElements()));
  }
  else {
    // Scalar case
    Type* unwrapped = newType.getUnwrappedLLVMType();
    if (newType.containsFloatingPointType())
      hasFloats = true;
    if (!unwrapped->isVoidTy())
      newType.setUnwrappedLLVMType(toScalarLLVMType(newType.getUnwrappedLLVMType()->getContext()));
  }
  return hasFloats;
}

ConversionScalarType& ConversionScalarType::operator=(const ConversionScalarType& other) {
  if (this == &other)
    return *this;
  ConversionType::operator=(other);
  sign = other.sign;
  bits = other.bits;
  fractionalBits = other.fractionalBits;
  floatStandard = other.floatStandard;
  return *this;
}

bool ConversionScalarType::operator==(const ConversionType& other) const {
  if (other.getKind() != K_Scalar)
    return false;
  auto& otherScalar = cast<ConversionScalarType>(other);
  return sign == otherScalar.sign && bits == otherScalar.bits && fractionalBits == otherScalar.fractionalBits
      && floatStandard == otherScalar.floatStandard;
}

std::unique_ptr<ConversionType> ConversionScalarType::clone(const TransparentType& type) const {
  auto copy = std::make_unique<ConversionScalarType>(*this);
  copy->transparentType = type.clone();
  return copy;
}

std::string ConversionScalarType::toString() const {
  std::stringstream ss;
  if (floatStandard == NotFloat)
    ss << (sign ? "s" : "u") << (bits - fractionalBits) << "_" << fractionalBits << "fixp";
  else
    ss << floatStandard << "flp";
  return ss.str();
}

ConversionStructType::ConversionStructType(const TransparentStructType& type,
                                           const std::shared_ptr<StructInfo>& structInfo,
                                           bool* conversionEnabled)
: ConversionType(type) {
  for (const auto&& [fieldType, fieldInfo] : zip(type.getFieldTypes(), *structInfo)) {
    if (!fieldInfo)
      fieldTypes.push_back(nullptr);
    else if (std::shared_ptr<ScalarInfo> scalarFieldInfo = std::dynamic_ptr_cast<ScalarInfo>(fieldInfo)) {
      if (scalarFieldInfo->isConversionEnabled())
        fieldTypes.push_back(std::make_unique<ConversionScalarType>(*fieldType, scalarFieldInfo->numericType.get()));
      else {
        if (conversionEnabled)
          *conversionEnabled = false;
        fieldTypes.push_back(nullptr);
      }
    }
    else if (std::shared_ptr<StructInfo> structFieldInfo = std::dynamic_ptr_cast<StructInfo>(fieldInfo)) {
      auto* structFieldType = cast<TransparentStructType>(fieldType);
      fieldTypes.push_back(
        std::make_unique<ConversionStructType>(*structFieldType, structFieldInfo, conversionEnabled));
    }
    else
      llvm_unreachable("unknown type of ValueInfo");
  }
}

bool ConversionStructType::toTransparentTypeHelper(TransparentType& newType) const {
  auto& newStructType = cast<TransparentStructType>(newType);
  assert(newStructType.getNumFieldTypes() == getNumFieldTypes());

  bool hasFloats = false;
  SmallVector<Type*, 4> fieldsLLVMTypes;
  for (unsigned i = 0; i < getNumFieldTypes(); i++) {
    TransparentType* fieldTransparentType = newStructType.getFieldType(i);
    ConversionType* fieldType = getFieldType(i);
    fieldsLLVMTypes.push_back(fieldTransparentType->getUnwrappedLLVMType());
    if (fieldTransparentType->isFloatingPointTyOrPtrTo() || fieldTransparentType->isStructTT())
      hasFloats |= fieldType->toTransparentTypeHelper(*fieldTransparentType);
  }
  if (hasFloats) {
    newStructType.setUnwrappedLLVMType(
      StructType::get(newStructType.getUnwrappedLLVMType()->getContext(),
                      fieldsLLVMTypes,
                      cast<StructType>(newStructType.getUnwrappedLLVMType())->isPacked()));
  }
  return hasFloats;
}

ConversionStructType& ConversionStructType::operator=(const ConversionStructType& other) {
  if (this == &other)
    return *this;
  ConversionType::operator=(other);
  fieldTypes.clear();
  fieldTypes.reserve(other.fieldTypes.size());
  for (const auto& ft : other.fieldTypes)
    fieldTypes.push_back(ft ? ft->clone() : nullptr);
  return *this;
}

bool ConversionStructType::operator==(const ConversionType& other) const {
  if (other.getKind() != K_Struct)
    return false;
  auto& otherStruct = cast<ConversionStructType>(other);
  if (fieldTypes.size() != otherStruct.fieldTypes.size())
    return false;
  for (size_t i = 0; i < fieldTypes.size(); i++)
    if (*fieldTypes[i] != *otherStruct.fieldTypes[i])
      return false;
  return true;
}

std::unique_ptr<ConversionType> ConversionStructType::clone(const TransparentType& type) const {
  auto copy = std::make_unique<ConversionStructType>(*this);
  copy->transparentType = type.clone();
  return copy;
}

std::string ConversionStructType::toString() const {
  std::stringstream ss;
  ss << '<';
  for (size_t i = 0; i < fieldTypes.size(); i++) {
    ConversionType* fieldType = fieldTypes[i].get();
    ss << (fieldType ? fieldType->toString() : "void");
    if (i != fieldTypes.size() - 1)
      ss << ',';
  }
  ss << '>';
  return ss.str();
}
