#ifndef TAFFO_NUMERIC_INFO_HPP
#define TAFFO_NUMERIC_INFO_HPP

#include "SerializationUtils.hpp"

#include <llvm/IR/Type.h>
#include <llvm/ADT/APFloat.h>

namespace taffo {

/// Info about a data type for numerical computations.
class NumericType : public Serializable, public Printable {
public:
  enum NumericTypeKind {
    K_FixpType,
    K_FloatType
  };

  NumericType(NumericTypeKind K) : Kind(K) {}
  virtual ~NumericType() {};

  virtual double getRoundingError() const = 0;
  /// Safe approximation of the minimum value representable with this Type.
  virtual llvm::APFloat getMinValueBound() const = 0;
  /// Safe approximation of the maximum value representable with this Type.
  virtual llvm::APFloat getMaxValueBound() const = 0;

  NumericTypeKind getKind() const { return Kind; }

  virtual bool operator==(const NumericType &other) const { return Kind == other.Kind; }
  virtual bool operator!=(const NumericType &other) const { return !(*this == other); }

  virtual std::shared_ptr<NumericType> clone() const = 0;

private:
  const NumericTypeKind Kind;
};

/// A Fixed Point Type.
/// Contains bit width, number of fractional bits of the format
/// and whether it is signed or not.
class FixpType : public NumericType {
public:
  static bool classof(const NumericType *T) { return T->getKind() == K_FixpType; }

  FixpType(unsigned Width, unsigned PointPos, bool Signed = true)
      : NumericType(K_FixpType), Width((Signed) ? -Width : Width), PointPos(PointPos) {}

  FixpType(int Width, unsigned PointPos)
      : NumericType(K_FixpType), Width(Width), PointPos(PointPos) {}

  double getRoundingError() const override;
  llvm::APFloat getMinValueBound() const override;
  llvm::APFloat getMaxValueBound() const override;

  unsigned int getWidth() const { return std::abs(Width); }
  int getSWidth() const { return Width; }
  unsigned int getPointPos() const { return PointPos; }
  bool isSigned() const { return Width < 0; }

  bool operator==(const NumericType &other) const override;

  std::shared_ptr<NumericType> clone() const override;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json &j) override;

private:
  int Width;         ///< Width of the format (in bits), negative if signed.
  unsigned PointPos; ///< Number of fractional bits.
};


/// A Floating Point Type.
/// Contains the particular type of floating point used, that must be supported by LLVM
class FloatType : public NumericType {
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

  static bool classof(const NumericType *T) { return T->getKind() == K_FloatType; }

  FloatType(FloatStandard standard, double greatestNumber)
      : NumericType(K_FloatType), standard(standard), greatestNumber(greatestNumber) {}

  FloatType(llvm::Type::TypeID TyId, double greatestNumber)
      : NumericType(K_FloatType), greatestNumber(greatestNumber) {
    switch (TyId) {
      case llvm::Type::HalfTyID: standard = Float_half; break;
      case llvm::Type::FloatTyID: standard = Float_float; break;
      case llvm::Type::DoubleTyID: standard = Float_double; break;
      case llvm::Type::FP128TyID: standard = Float_fp128; break;
      case llvm::Type::X86_FP80TyID: standard = Float_x86_fp80; break;
      case llvm::Type::PPC_FP128TyID: standard = Float_ppc_fp128; break;
      case llvm::Type::BFloatTyID: standard = Float_bfloat; break;
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

  bool operator==(const NumericType &other) const override;

  std::shared_ptr<NumericType> clone() const override;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json &j) override;

protected:
  FloatStandard standard;

  // This is only used to understand the maximum error that this type can generate
  // As during the DTA pass we assign each Type looking at its range, it is "free" (as in free beer)
  double greatestNumber;
};

} // namespace taffo

#endif // TAFFO_NUMERIC_INFO_HPP
