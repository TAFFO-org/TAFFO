#pragma once

#include "TaffoInfo/ValueInfo.hpp"
#include "Types/TransparentType.hpp"

#include <string>

namespace taffo {

class FixedPointType : public Printable {
public:
  friend class FixedPointScalarType;
  friend class FixedPointStructType;

  enum FixedPointTypeKind {
    K_Scalar,
    K_Struct
  };

  virtual ~FixedPointType() = default;

  virtual bool isInvalid() const = 0;
  virtual bool isFixedPoint() const { return false; }
  virtual bool isFloatingPoint() const { return false; }
  std::shared_ptr<TransparentType> toTransparentType(const std::shared_ptr<TransparentType>& srcType,
                                                     bool* hasFloats = nullptr) const;
  std::shared_ptr<FixedPointType> unwrapIndexList(const std::shared_ptr<TransparentType>& srcType,
                                                  llvm::ArrayRef<unsigned> indices);
  std::shared_ptr<FixedPointType> unwrapIndexList(const std::shared_ptr<TransparentType>& srcType,
                                                  llvm::iterator_range<const llvm::Use*> indices);
  virtual FixedPointTypeKind getKind() const = 0;

  virtual bool operator==(const FixedPointType& other) const = 0;
  bool operator!=(const FixedPointType& other) const { return !(*this == other); }

  virtual std::shared_ptr<FixedPointType> clone() const = 0;

protected:
  virtual bool toTransparentTypeHelper(const std::shared_ptr<TransparentType>& newType) const = 0;
};

class FixedPointScalarType : public FixedPointType {
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

  static bool classof(const FixedPointType* type) { return type->getKind() == K_Scalar; }

  FixedPointScalarType();
  FixedPointScalarType(bool isSigned, int bits, int fractionalBits);
  FixedPointScalarType(llvm::Type* type, bool isSigned = true);
  FixedPointScalarType(NumericTypeInfo* numericType);
  FixedPointScalarType(const FixedPointScalarType& other);

  bool isSigned() const { return sign; }
  void setSigned(bool isSigned) { this->sign = isSigned; }
  int getBits() const { return bits; }
  void setBits(int bits) { this->bits = bits; }
  int getFractionalBits() const { return fractionalBits; }
  void setFractionalBits(int fractionalBits) { this->fractionalBits = fractionalBits; }
  int getIntegerBits() const { return bits - fractionalBits; }
  FloatStandard getFloatStandard() const { return floatStandard; }

  bool isInvalid() const override;
  bool isFixedPoint() const override { return floatStandard == NotFloat; }
  bool isFloatingPoint() const override { return floatStandard != NotFloat; }
  llvm::Type* scalarToLLVMType(llvm::LLVMContext& context) const;
  FixedPointTypeKind getKind() const override { return K_Scalar; }

  bool operator==(const FixedPointType& other) const override;

  std::shared_ptr<FixedPointType> clone() const override;
  std::string toString() const override;

private:
  bool sign;
  int bits;
  int fractionalBits;
  FloatStandard floatStandard;

protected:
  bool toTransparentTypeHelper(const std::shared_ptr<TransparentType>& newType) const override;
};

class FixedPointStructType : public FixedPointType {
public:
  static bool classof(const FixedPointType* type) { return type->getKind() == K_Struct; }

  FixedPointStructType(const llvm::ArrayRef<std::shared_ptr<FixedPointType>>& fields);
  FixedPointStructType(const std::shared_ptr<StructInfo>& structInfo, int* enableConversion);
  FixedPointStructType(const FixedPointStructType& other);

  size_t getNumFieldTypes() const { return fieldTypes.size(); }
  std::shared_ptr<FixedPointType> getFieldType(unsigned i) const { return fieldTypes[i]; }

  bool isInvalid() const override;
  FixedPointTypeKind getKind() const override { return K_Struct; }

  bool operator==(const FixedPointType& other) const override;

  std::shared_ptr<FixedPointType> clone() const override;
  std::string toString() const override;

private:
  llvm::SmallVector<std::shared_ptr<FixedPointType>, 4> fieldTypes;

protected:
  bool toTransparentTypeHelper(const std::shared_ptr<TransparentType>& newType) const override;
};

} // namespace taffo
