#pragma once

#include "TaffoInfo/ValueInfo.hpp"
#include "TransparentType.hpp"

#include <string>

namespace taffo {

class ConversionType;

class ConversionTypeFactory {
public:
  static std::unique_ptr<ConversionType> create(const tda::TransparentType& type);
};

class ConversionType : public tda::Printable {
public:
  friend class ConversionScalarType;
  friend class ConversionStructType;

  enum ConversionTypeKind {
    K_Scalar,
    K_Struct
  };

  virtual ConversionTypeKind getKind() const = 0;

  ConversionType(const tda::TransparentType& type)
  : transparentType(type.clone()) {}

  ConversionType(const ConversionType& other)
  : transparentType(other.transparentType->clone()) {}

  virtual ~ConversionType() = default;

  bool isVoid() const { return transparentType->getLLVMType()->isVoidTy(); }
  bool isPtr() const { return transparentType->isPointerTT(); }
  bool isOpaquePtr() const { return transparentType->isOpaquePtr(); }
  virtual bool isFixedPoint() const { return false; }
  virtual bool isFloatingPoint() const { return false; }

  tda::TransparentType* toTransparentType(bool* hasFloats = nullptr) const;
  llvm::Type* toLLVMType(bool* hasFloats = nullptr) const { return toTransparentType(hasFloats)->toLLVMType(); }

  std::unique_ptr<ConversionType> getGepConvType(llvm::ArrayRef<unsigned> gepIndices) const;
  std::unique_ptr<ConversionType> getGepConvType(llvm::iterator_range<const llvm::Use*> gepIndices) const;

  ConversionType& operator=(const ConversionType& other);
  virtual bool operator==(const ConversionType& other) const = 0;
  bool operator!=(const ConversionType& other) const { return !(*this == other); }

  std::unique_ptr<ConversionType> clone() const { return clone(*transparentType); }
  virtual std::unique_ptr<ConversionType> clone(const tda::TransparentType& type) const = 0;

protected:
  std::unique_ptr<tda::TransparentType> transparentType;
  mutable bool recomputedTransparentType = false;
  mutable bool hasFloats = false;

  virtual bool toTransparentTypeHelper(tda::TransparentType& newType) const = 0;
};

class ConversionScalarType : public ConversionType {
public:
  enum FloatStandard {
    NotFloat = -1,
    Float_half = 0,
    Float_float,
    Float_double,
    Float_fp128,
    Float_x86_fp80,
    Float_ppc_fp128,
    Float_bfloat
  };

  static bool classof(const ConversionType* type) { return type->getKind() == K_Scalar; }

  ConversionTypeKind getKind() const override { return K_Scalar; }

  ConversionScalarType(const tda::TransparentType& type, bool isSigned, int bits, int fractionalBits)
  : ConversionType(type), sign(isSigned), bits(bits), fractionalBits(fractionalBits), floatStandard(NotFloat) {}

  ConversionScalarType(const tda::TransparentType& type, bool isSigned = true);
  ConversionScalarType(const tda::TransparentType& type, NumericTypeInfo* numericType);

  ConversionScalarType(const ConversionScalarType& other)
  : ConversionType(other),
    sign(other.sign),
    bits(other.bits),
    fractionalBits(other.fractionalBits),
    floatStandard(other.floatStandard) {}

  bool isSigned() const { return sign; }
  void setSigned(bool isSigned) { this->sign = isSigned; }
  int getBits() const { return bits; }
  void setBits(int bits) { this->bits = bits; }
  int getFractionalBits() const { return fractionalBits; }
  void setFractionalBits(int fractionalBits) { this->fractionalBits = fractionalBits; }
  int getIntegerBits() const { return bits - fractionalBits; }
  FloatStandard getFloatStandard() const { return floatStandard; }

  bool isFixedPoint() const override { return floatStandard == NotFloat; }
  bool isFloatingPoint() const override { return floatStandard != NotFloat; }
  llvm::Type* toScalarLLVMType(llvm::LLVMContext& context) const;

  ConversionScalarType& operator=(const ConversionScalarType& other);
  bool operator==(const ConversionType& other) const override;

  using ConversionType::clone;
  std::unique_ptr<ConversionType> clone(const tda::TransparentType& type) const override;
  std::string toString() const override;

private:
  bool sign;
  int bits;
  int fractionalBits;
  FloatStandard floatStandard;

protected:
  bool toTransparentTypeHelper(tda::TransparentType& newType) const override;
};

class ConversionStructType : public ConversionType {
public:
  static bool classof(const ConversionType* type) { return type->getKind() == K_Struct; }

  ConversionTypeKind getKind() const override { return K_Struct; }

  ConversionStructType(const tda::TransparentType& type, const llvm::ArrayRef<std::unique_ptr<ConversionType>>& fields)
  : ConversionType(type) {
    assert(type.isStructTTOrPtrTo());
    for (const auto& field : fields)
      fieldTypes.push_back(field ? field->clone() : nullptr);
  }

  ConversionStructType(const tda::TransparentType& type,
                       const std::shared_ptr<StructInfo>& structInfo,
                       bool* conversionEnabled);

  ConversionStructType(const ConversionStructType& other)
  : ConversionType(other) {
    for (const auto& field : other.fieldTypes)
      fieldTypes.push_back(field ? field->clone() : nullptr);
  }

  size_t getNumFieldTypes() const { return fieldTypes.size(); }
  ConversionType* getFieldType(unsigned i) const { return fieldTypes[i].get(); }

  ConversionStructType& operator=(const ConversionStructType& other);
  bool operator==(const ConversionType& other) const override;

  using ConversionType::clone;
  std::unique_ptr<ConversionType> clone(const tda::TransparentType& type) const override;
  std::string toString() const override;

private:
  llvm::SmallVector<std::unique_ptr<ConversionType>, 4> fieldTypes;

protected:
  bool toTransparentTypeHelper(tda::TransparentType& newType) const override;
};

} // namespace taffo
