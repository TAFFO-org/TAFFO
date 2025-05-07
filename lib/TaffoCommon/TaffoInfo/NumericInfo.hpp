#pragma once

#include "SerializationUtils.hpp"

#include <llvm/ADT/APFloat.h>
#include <llvm/IR/Type.h>

namespace taffo {

class NumericTypeInfo : public Serializable,
                        public Printable {
public:
  enum NumericTypeKind {
    K_FixedPoint,
    K_FloatingPoint
  };

  virtual ~NumericTypeInfo() {}

  virtual double getRoundingError() const = 0;
  /// Safe approximation of the minimum value representable with this Type.
  virtual llvm::APFloat getMinValueBound() const = 0;
  /// Safe approximation of the maximum value representable with this Type.
  virtual llvm::APFloat getMaxValueBound() const = 0;

  virtual NumericTypeKind getKind() const = 0;

  virtual bool operator==(const NumericTypeInfo& other) const { return getKind() == other.getKind(); }
  virtual bool operator!=(const NumericTypeInfo& other) const { return !(*this == other); }

  virtual std::shared_ptr<NumericTypeInfo> clone() const = 0;
};

class FixedPointInfo : public NumericTypeInfo {
public:
  static bool classof(const NumericTypeInfo* T) { return T->getKind() == K_FixedPoint; }

  FixedPointInfo(bool isSigned, unsigned bits, unsigned fractionalBits)
  : sign(isSigned), bits(bits), fractionalBits(fractionalBits) {}

  double getRoundingError() const override;
  llvm::APFloat getMinValueBound() const override;
  llvm::APFloat getMaxValueBound() const override;

  bool isSigned() const { return sign; }
  unsigned getBits() const { return bits; }
  unsigned getFractionalBits() const { return fractionalBits; }
  NumericTypeKind getKind() const override { return K_FixedPoint; }

  bool operator==(const NumericTypeInfo& other) const override;

  std::shared_ptr<NumericTypeInfo> clone() const override;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json& j) override;

private:
  bool sign;
  unsigned bits;
  unsigned fractionalBits;
};

class FloatingPointInfo : public NumericTypeInfo {
public:
  enum FloatStandard {
    Float_half = 0,  /*16-bit floating-point value*/
    Float_float,     /*32-bit floating-point value*/
    Float_double,    /*64-bit floating-point value*/
    Float_fp128,     /*128-bit floating-point value (112-bit mantissa)*/
    Float_x86_fp80,  /*80-bit floating-point value (X87)*/
    Float_ppc_fp128, /*128-bit floating-point value (two 64-bits)*/
    Float_bfloat
  };

  static std::string getFloatStandardName(FloatStandard standard);

  static bool classof(const NumericTypeInfo* T) { return T->getKind() == K_FloatingPoint; }

  FloatingPointInfo(FloatStandard standard, double greatestNumber)
  : standard(standard), greatestNumber(greatestNumber) {}

  FloatingPointInfo(llvm::Type::TypeID typeId, double greatestNumber)
  : greatestNumber(greatestNumber) {
    switch (typeId) {
    case llvm::Type::HalfTyID:
      standard = Float_half;
      break;
    case llvm::Type::FloatTyID:
      standard = Float_float;
      break;
    case llvm::Type::DoubleTyID:
      standard = Float_double;
      break;
    case llvm::Type::FP128TyID:
      standard = Float_fp128;
      break;
    case llvm::Type::X86_FP80TyID:
      standard = Float_x86_fp80;
      break;
    case llvm::Type::PPC_FP128TyID:
      standard = Float_ppc_fp128;
      break;
    case llvm::Type::BFloatTyID:
      standard = Float_bfloat;
      break;
    default:
      llvm_unreachable("invalid type id for FloatType's constructor");
    }
  }

  double getRoundingError() const override;
  llvm::APFloat getMinValueBound() const override;
  llvm::APFloat getMaxValueBound() const override;

  int getP() const;
  llvm::Type::TypeID getLLVMTypeID() const;
  FloatStandard getStandard() const { return standard; }
  double getGreatestNumber() const { return greatestNumber; }
  NumericTypeKind getKind() const override { return K_FloatingPoint; }

  bool operator==(const NumericTypeInfo& other) const override;

  std::shared_ptr<NumericTypeInfo> clone() const override;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json& j) override;

protected:
  FloatStandard standard;

  // This is only used to understand the maximum error that this type can generate
  // As during the DTA pass we assign each Type looking at its range, it is "free" (as in free beer)
  double greatestNumber;
};

} // namespace taffo
