#include "ConversionType.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TransparentType.hpp"
#include "Utils/PtrCasts.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/Casting.h>

#include <memory>
#include <sstream>

#define DEBUG_TYPE "taffo-conv"

using namespace llvm;
using namespace tda;
using namespace taffo;

std::unique_ptr<ConversionType> ConversionTypeFactory::create(const TransparentType& type) {
  if (type.isStructTTOrPtrTo()) {
    auto* structType = cast<TransparentStructType>(type.getFirstNonPtr());
    SmallVector<std::unique_ptr<ConversionType>, 4> fields;
    fields.reserve(structType->getNumFieldTypes());
    for (unsigned i = 0; i < structType->getNumFieldTypes(); i++)
      fields.push_back(create(*structType->getFieldType(i)));
    return std::make_unique<ConversionStructType>(type, fields);
  }
  return std::make_unique<ConversionScalarType>(type);
}

const TransparentType* ConversionType::toTransparentType(bool* hasFloats) const {
  if (!recomputedTransparentType) {
    recomputedTransparentType = true;
    this->hasFloats = toTransparentTypeHelper(*transparentType);
  }
  if (hasFloats)
    *hasFloats = this->hasFloats;
  return transparentType;
}

std::unique_ptr<ConversionType> ConversionType::getGepConvType(const ArrayRef<unsigned> gepIndices) const {
  const TransparentType* resolvedType = transparentType;
  const ConversionType* resolvedConvType = this;
  for (unsigned index : gepIndices)
    if (resolvedType->isPointerTT())
      resolvedType = resolvedType->getPointedType();
    else if (resolvedType->isArrayTT())
      resolvedType = cast<TransparentArrayType>(resolvedType)->getElementType();
    else if (resolvedType->isStructTT()) {
      resolvedType = cast<TransparentStructType>(resolvedType)->getFieldType(index);
      resolvedConvType = cast<ConversionStructType>(resolvedConvType)->getFieldType(index);
    }
    else
      llvm_unreachable("Unsupported type in gep");

  if (!resolvedConvType)
    return nullptr;
  return resolvedConvType->clone(*TransparentPointerType::get(resolvedType->getLLVMContext(), resolvedType));
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
  return getGepConvType(ArrayRef<unsigned> {indicesVector});
}

ConversionType& ConversionType::operator=(const ConversionType& other) {
  if (this == &other)
    return *this;
  transparentType = other.transparentType ? other.transparentType : nullptr;
  recomputedTransparentType = other.recomputedTransparentType;
  hasFloats = other.hasFloats;
  return *this;
}

ConversionScalarType::ConversionScalarType(const TransparentType& type, bool isSigned)
: ConversionType(type), sign(isSigned) {
  assert(!type.isStructTTOrPtrTo());
  const TransparentType* curr = &type;
  while (curr->isArrayTT() || (curr->isPointerTT() && !curr->isOpaquePtr()))
    if (const auto* currArray = dyn_cast<TransparentArrayType>(curr))
      curr = currArray->getElementType();
    else if (const auto* currPointer = dyn_cast<TransparentPointerType>(curr))
      curr = currPointer->getPointedType();
  const Type* unwrappedType = curr->getLLVMType();
  if (unwrappedType && unwrappedType->isFloatingPointTy()) {
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
  else if (unwrappedType && unwrappedType->isIntegerTy()) {
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
  if (isVoid())
    return Type::getVoidTy(context);
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

bool ConversionScalarType::toTransparentTypeHelper(const TransparentType& newType) const {
  // Pointer case
  if (newType.isPointerTT()) {
    if (newType.isOpaquePtr())
      return false;

    auto tempConv = std::make_unique<ConversionScalarType>(*this);
    if (tempConv->toTransparentTypeHelper(*newType.getPointedType())) {
      const TransparentType* tempConvType = tempConv->transparentType;
      transparentType = TransparentPointerType::get(tempConvType->getLLVMContext(), tempConvType);
      return true;
    }
    return false;
  }

  // Array case
  if (newType.isArrayTT()) {
    const auto& newArrayType = cast<TransparentArrayType>(newType);

    auto elementConv = std::make_unique<ConversionScalarType>(*this);
    if (elementConv->toTransparentTypeHelper(*(newArrayType.getElementType()))) {
      transparentType = newArrayType.setElementType(elementConv->transparentType);
      return true;
    }
    return false;
  }

  // Pure Scalar case
  else {
    const Type* unwrappedLLVMType = newType.getLLVMType();
    bool localHasFloats = false;

    if (newType.containsFloatingPointType())
      localHasFloats = true;

    if (!unwrappedLLVMType->isVoidTy()) {
      const llvm::Type* targetLLVMType = toScalarLLVMType(unwrappedLLVMType->getContext());
      if (unwrappedLLVMType != targetLLVMType) {
        transparentType = TransparentType::get(newType.getLLVMContext(), targetLLVMType);
        if (transparentType->containsFloatingPointType())
          localHasFloats = true;
      }
    }

    return localHasFloats;
  }
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
  copy->transparentType = &type;
  return copy;
}

std::string ConversionScalarType::toString() const {
  std::stringstream ss;
  if (isVoid())
    ss << "void";
  else if (isOpaquePtr())
    ss << "ptr";
  else if (floatStandard == NotFloat)
    ss << (sign ? "s" : "u") << (bits - fractionalBits) << "_" << fractionalBits << "fixp";
  else
    ss << floatStandard << "flp";
  return ss.str();
}

ConversionStructType::ConversionStructType(const TransparentType& type,
                                           const std::shared_ptr<StructInfo>& structInfo,
                                           bool* conversionEnabled)
: ConversionType(type) {
  if (conversionEnabled)
    *conversionEnabled = false;
  for (const auto&& [fieldType, fieldInfo] :
       zip(cast<TransparentStructType>(type.getFirstNonPtr())->getFieldTypes(), *structInfo)) {
    if (!fieldInfo)
      fieldTypes.push_back(ConversionTypeFactory::create(*fieldType));
    else if (std::shared_ptr<ScalarInfo> scalarFieldInfo = std::dynamic_ptr_cast<ScalarInfo>(fieldInfo)) {
      if (scalarFieldInfo->isConversionEnabled() && scalarFieldInfo->numericType) {
        if (conversionEnabled)
          *conversionEnabled = true;
        fieldTypes.push_back(std::make_unique<ConversionScalarType>(*fieldType, scalarFieldInfo->numericType.get()));
      }
      else
        fieldTypes.push_back(ConversionTypeFactory::create(*fieldType));
    }
    else if (std::shared_ptr<StructInfo> structFieldInfo = std::dynamic_ptr_cast<StructInfo>(fieldInfo)) {
      auto* structFieldType = cast<TransparentStructType>(fieldType->getFirstNonPtr());
      bool structFieldConversionEnabled;
      fieldTypes.push_back(
        std::make_unique<ConversionStructType>(*structFieldType, structFieldInfo, &structFieldConversionEnabled));
      if (conversionEnabled && structFieldConversionEnabled)
        *conversionEnabled = true;
    }
    else
      llvm_unreachable("unknown type of valueInfo");
  }
}

bool ConversionStructType::toTransparentTypeHelper(const TransparentType& newType) const {
  if (newType.isPointerTT()) {
    const TransparentType* pointed = newType.getPointedType();
    if (toTransparentTypeHelper(*pointed)) {
      transparentType = TransparentPointerType::get(transparentType->getLLVMContext(), transparentType);
      return true;
    }
    return false;
  }

  const auto& newStructType = cast<TransparentStructType>(newType);
  assert(newStructType.getNumFieldTypes() == getNumFieldTypes());
  bool hasFloats = false;
  SmallVector<const TransparentType*, 8> fieldTypes;
  SmallVector<Type*, 8> fieldLLVMTypes;
  for (unsigned i = 0; i < getNumFieldTypes(); ++i) {
    const TransparentType* fieldTransparentType = newStructType.getFieldType(i);
    ConversionType* fieldConvType = getFieldType(i);
    if (fieldConvType)
      if (fieldTransparentType->isFloatingPointTyOrPtrTo() || fieldTransparentType->isStructTTOrPtrTo()) {
        if (fieldConvType->toTransparentTypeHelper(*fieldTransparentType)) {
          hasFloats = true;
          fieldTransparentType = fieldConvType->transparentType;
        }
      }
    fieldTypes.push_back(fieldTransparentType);
    fieldLLVMTypes.push_back(const_cast<Type*>(fieldTransparentType->getLLVMType()));
  }

  if (hasFloats) {
    const Type* llvmType = StructType::get(newStructType.getLLVMType()->getContext(),
                                           fieldLLVMTypes,
                                           cast<StructType>(newStructType.getLLVMType())->isPacked());
    const TransparentType* result = TransparentStructType::get(newStructType.getLLVMContext(), fieldTypes, llvmType);
    transparentType = result;
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
  for (size_t i = 0; i < fieldTypes.size(); i++) {
    ConversionType* fieldType = fieldTypes[i].get();
    ConversionType* otherFieldType = otherStruct.fieldTypes[i].get();
    if ((!fieldType && otherFieldType) || (!otherFieldType && fieldType))
      return false;
    if (fieldType && otherFieldType && *fieldType != *otherFieldType)
      return false;
  }
  return true;
}

std::unique_ptr<ConversionType> ConversionStructType::clone(const TransparentType& type) const {
  auto copy = std::make_unique<ConversionStructType>(*this);
  copy->transparentType = &type;
  return copy;
}

std::string ConversionStructType::toString() const {
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
