#include "ConversionPass.hpp"
#include "FixedPointType.hpp"
#include "PtrCasts.hpp"
#include "Types/TransparentType.hpp"

#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/Casting.h>

#include <memory>
#include <sstream>

#define DEBUG_TYPE "taffo-conversion"

using namespace llvm;
using namespace taffo;
using namespace taffo;

std::shared_ptr<TransparentType> FixedPointType::toTransparentType(const std::shared_ptr<TransparentType>& srcType,
                                                                   bool* hasFloats) const {
  std::shared_ptr<TransparentType> newType = srcType->clone();
  bool floats = toTransparentTypeHelper(newType);
  if (hasFloats)
    *hasFloats = floats;
  return newType;
}

std::shared_ptr<FixedPointType> FixedPointType::unwrapIndexList(const std::shared_ptr<TransparentType>& srcType,
                                                                ArrayRef<unsigned> indices) {
  std::shared_ptr<TransparentType> resolvedType = srcType;
  std::shared_ptr<FixedPointType> resolvedFixpType = this->clone();
  for (unsigned index : indices)
    if (resolvedType->isPointerType())
      resolvedType = resolvedType->getPointedType();
    else if (resolvedType->isStructType()) {
      resolvedType = std::static_ptr_cast<TransparentStructType>(resolvedType)->getFieldType(index);
      resolvedFixpType = std::static_ptr_cast<FixedPointStructType>(resolvedFixpType)->getFieldType(index);
    }
    else if (resolvedType->isArrayType())
      resolvedType = std::static_ptr_cast<TransparentArrayType>(resolvedType)->getArrayElementType();
    else
      llvm_unreachable("Unsupported type in GEP");
  return resolvedFixpType;
}

std::shared_ptr<FixedPointType> FixedPointType::unwrapIndexList(const std::shared_ptr<TransparentType>& srcType,
                                                                iterator_range<const Use*> indices) {
  SmallVector<unsigned, 4> indicesVector;
  for (Value* val : indices) {
    auto constantIndex = dyn_cast<ConstantInt>(val);
    // The constant value of the index is only used to navigate struct types.
    // In other cases indicesVector is only used to count pointer indirections,
    // so only its cardinality matters and not the values themselves
    indicesVector.push_back(constantIndex ? constantIndex->getZExtValue() : 0);
  }
  return unwrapIndexList(srcType, indicesVector);
}

FixedPointScalarType::FixedPointScalarType()
: sign(false), bits(0), fractionalBits(0), floatStandard(NotFloat) {}

FixedPointScalarType::FixedPointScalarType(bool isSigned, int bits, int fractionalBits)
: sign(isSigned), bits(bits), fractionalBits(fractionalBits), floatStandard(NotFloat) {}

FixedPointScalarType::FixedPointScalarType(Type* type, bool isSigned)
: sign(isSigned) {
  if (type->isFloatingPointTy()) {
    bits = 0;
    fractionalBits = 0;
    if (type->getTypeID() == Type::HalfTyID)
      floatStandard = Float_half;
    else if (type->getTypeID() == Type::DoubleTyID)
      floatStandard = Float_double;
    else if (type->getTypeID() == Type::FloatTyID)
      floatStandard = Float_float;
    else if (type->getTypeID() == Type::FP128TyID)
      floatStandard = Float_fp128;
    else if (type->getTypeID() == Type::PPC_FP128TyID)
      floatStandard = Float_ppc_fp128;
    else if (type->getTypeID() == Type::X86_FP80TyID)
      floatStandard = Float_x86_fp80;
    else if (type->getTypeID() == Type::BFloatTyID)
      floatStandard = Float_bfloat;
    else
      floatStandard = NotFloat;
  }
  else if (type->isIntegerTy()) {
    bits = type->getIntegerBitWidth();
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

FixedPointScalarType::FixedPointScalarType(NumericTypeInfo* numericType) {
  if (numericType) {
    if (auto* fpt = dyn_cast<FixedPointInfo>(numericType)) {
      bits = fpt->getBits();
      fractionalBits = fpt->getFractionalBits();
      sign = fpt->isSigned();
      floatStandard = NotFloat;
    }
    else if (auto* flt = dyn_cast<FloatingPointInfo>(numericType)) {
      bits = 0;
      fractionalBits = 0;
      sign = true;
      floatStandard = static_cast<FloatStandard>(flt->getStandard());
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

FixedPointScalarType::FixedPointScalarType(const FixedPointScalarType& other)
: sign(other.sign), bits(other.bits), fractionalBits(other.fractionalBits), floatStandard(other.floatStandard) {}

bool FixedPointScalarType::isInvalid() const { return bits == 0 && floatStandard == NotFloat; }

Type* FixedPointScalarType::scalarToLLVMType(LLVMContext& context) const {
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
  default:
    llvm_unreachable("Unhandled floating point type");
  }
}

bool FixedPointScalarType::toTransparentTypeHelper(const std::shared_ptr<TransparentType>& newType) const {
  if (newType->isArrayType()) {
    // Array Case
    std::shared_ptr<TransparentArrayType> arrType = std::dynamic_ptr_cast<TransparentArrayType>(newType);
    toTransparentTypeHelper(arrType->getArrayElementType());
    newType->unwrappedType =
      ArrayType::get(arrType->getArrayElementType()->getUnwrappedType(), newType->unwrappedType->getArrayNumElements());
  }
  else {
    // Scalar Case
    Type* unwrapped = newType->getUnwrappedType();
    if (!unwrapped->isVoidTy())
      newType->unwrappedType = scalarToLLVMType(newType->getUnwrappedType()->getContext());
    return true;
  }
  return false;
}

bool FixedPointScalarType::operator==(const FixedPointType& other) const {
  if (other.getKind() != K_Scalar)
    return false;
  auto& otherScalar = cast<FixedPointScalarType>(other);
  return sign == otherScalar.sign && bits == otherScalar.bits && fractionalBits == otherScalar.fractionalBits
      && floatStandard == otherScalar.floatStandard;
}

std::shared_ptr<FixedPointType> FixedPointScalarType::clone() const {
  return std::make_shared<FixedPointScalarType>(*this);
}

std::string FixedPointScalarType::toString() const {
  std::stringstream ss;
  if (floatStandard == NotFloat)
    ss << (sign ? "s" : "u") << (bits - fractionalBits) << "_" << fractionalBits << "fixp";
  else
    ss << floatStandard << "flp";
  return ss.str();
}

FixedPointStructType::FixedPointStructType(const ArrayRef<std::shared_ptr<FixedPointType>>& fields)
: fieldTypes(fields) {}

FixedPointStructType::FixedPointStructType(const std::shared_ptr<StructInfo>& structInfo, int* enableConversion) {
  for (const std::shared_ptr<ValueInfo>& fieldInfo : *structInfo) {
    if (!fieldInfo)
      fieldTypes.push_back(std::make_shared<FixedPointScalarType>());
    else if (std::shared_ptr<ScalarInfo> scalarFieldInfo = std::dynamic_ptr_cast<ScalarInfo>(fieldInfo)) {
      if (scalarFieldInfo->isConversionEnabled()) {
        if (enableConversion)
          (*enableConversion)++;
        fieldTypes.push_back(std::make_shared<FixedPointScalarType>(scalarFieldInfo->numericType.get()));
      }
      else
        fieldTypes.push_back(std::make_shared<FixedPointScalarType>());
    }
    else if (std::shared_ptr<StructInfo> structFieldInfo = std::dynamic_ptr_cast<StructInfo>(fieldInfo))
      fieldTypes.push_back(std::make_shared<FixedPointStructType>(structFieldInfo, enableConversion));
    else
      llvm_unreachable("unknown type of ValueInfo");
  }
}

FixedPointStructType::FixedPointStructType(const FixedPointStructType& other)
: fieldTypes(other.fieldTypes) {}

bool FixedPointStructType::isInvalid() const {
  for (const auto& fpt : fieldTypes)
    if (fpt->isInvalid())
      return true;
  return false;
}

bool FixedPointStructType::toTransparentTypeHelper(const std::shared_ptr<TransparentType>& newType) const {
  std::shared_ptr<TransparentStructType> newStructType = std::static_ptr_cast<TransparentStructType>(newType);
  assert(newStructType->getNumFieldTypes() == getNumFieldTypes());

  bool hasFloats = false;
  SmallVector<Type*, 4> fieldsLLVMTypes;
  for (unsigned i = 0; i < getNumFieldTypes(); i++) {
    std::shared_ptr<TransparentType> fieldTransparentType = newStructType->getFieldType(i);
    std::shared_ptr<FixedPointType> fieldType = getFieldType(i);
    fieldsLLVMTypes.push_back(fieldTransparentType->getUnwrappedType());
    if (fieldType->isInvalid())
      continue;
    if (fieldTransparentType->isFloatingPointType() || fieldTransparentType->isStructType())
      hasFloats |= fieldType->toTransparentTypeHelper(fieldTransparentType);
  }
  if (hasFloats) {
    newStructType->unwrappedType = StructType::get(newStructType->getUnwrappedType()->getContext(),
                                                   fieldsLLVMTypes,
                                                   cast<StructType>(newStructType->getUnwrappedType())->isPacked());
  }
  return hasFloats;
}

bool FixedPointStructType::operator==(const FixedPointType& other) const {
  if (other.getKind() != K_Struct)
    return false;
  auto& otherStruct = cast<FixedPointStructType>(other);
  if (fieldTypes.size() != otherStruct.fieldTypes.size())
    return false;
  for (size_t i = 0; i < fieldTypes.size(); i++)
    if (*fieldTypes[i] != *otherStruct.fieldTypes[i])
      return false;
  return true;
}

std::shared_ptr<FixedPointType> FixedPointStructType::clone() const {
  return std::make_shared<FixedPointStructType>(*this);
}

std::string FixedPointStructType::toString() const {
  std::stringstream ss;
  ss << '<';
  for (size_t i = 0; i < fieldTypes.size(); i++) {
    ss << fieldTypes[i]->toString();
    if (i != fieldTypes.size() - 1)
      ss << ',';
  }
  ss << '>';
  return ss.str();
}
