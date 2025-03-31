#ifndef __FIXED_POINT_TYPE_H__
#define __FIXED_POINT_TYPE_H__

#include "TaffoInfo/ValueInfo.hpp"

#include <llvm/ADT/Statistic.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <fstream>

#define DEBUG_TYPE "taffo-conversion"

namespace flttofix {

class FixedPointType {
public:
  enum FloatStandard {
    Float_NotFloat = -1,
    Float_half = 0,  /*16-bit floating-point value*/
    Float_float,     /*32-bit floating-point value*/
    Float_double,    /*64-bit floating-point value*/
    Float_fp128,     /*128-bit floating-point value (112-bit mantissa)*/
    Float_x86_fp80,  /*80-bit floating-point value (X87)*/
    Float_ppc_fp128, /*128-bit floating-point value (two 64-bits)*/
    Float_bfloat
  };

private:
  struct Primitive {
    bool isSigned;
    int fracBitsAmt;
    int bitsAmt;

    FloatStandard floatStandard;


    bool operator==(const Primitive &rhs) const
    {
      return this->isSigned == rhs.isSigned &&
             this->fracBitsAmt == rhs.fracBitsAmt &&
             this->bitsAmt == rhs.bitsAmt &&
             this->floatStandard == rhs.floatStandard;
    };

    std::string toString() const;
  };

  std::shared_ptr<llvm::SmallVector<FixedPointType, 2>> structData;
  Primitive scalarData;

public:
  /** Default scalar type (invalid 0/0 parameters default) */
  FixedPointType();

  /** Scalar type (also used for arrays)
   *  @param s true for signed types, false for unsigned types
   *  @param f Size of the fractional part in bits
   *  @param b Size of the type in bits */
  FixedPointType(bool s, int f, int b);

  /** Struct type
   *  @param elems List of types, one for each struct field. Use a type with
   *    zero bitsAmt for non-fixed-point elements */
  FixedPointType(const llvm::ArrayRef<FixedPointType> &elems);

  /** Scalar type from integer type (invalid when llvmtype is a float)
   *  @param llvmtype An integer type
   *  @param signd If the resulting fixed point type is signed */
  FixedPointType(llvm::Type *llvmtype, bool signd = true);

  FixedPointType(taffo::NumericType *mdtype);

  static FixedPointType get(taffo::ValueInfo *mdnfo, int *enableConversion = nullptr);

  std::string toString() const;

  llvm::Type *scalarToLLVMType(llvm::LLVMContext &ctxt) const;
  llvm::Type *toLLVMType(llvm::Type *srct, bool *hasfloats) const;

  bool &scalarIsSigned()
  {
    assert(!structData && "fixed point type not a scalar");
    return scalarData.isSigned;
  };

  bool scalarIsSigned() const
  {
    assert(!structData && "fixed point type not a scalar");
    return scalarData.isSigned;
  };

  int &scalarFracBitsAmt()
  {
    assert(!structData && "fixed point type not a scalar");
    assert(scalarData.floatStandard == Float_NotFloat && "this type is a float!");
    return scalarData.fracBitsAmt;
  };

  int scalarFracBitsAmt() const
  {
    assert(!structData && "fixed point type not a scalar");
    assert(scalarData.floatStandard == Float_NotFloat && "this type is a float!");
    return scalarData.fracBitsAmt;
  };

  int &scalarBitsAmt()
  {
    assert(!structData && "fixed point type not a scalar");
    assert(scalarData.floatStandard == Float_NotFloat && "this type is a float!");
    return scalarData.bitsAmt;
  };

  int scalarBitsAmt() const
  {
    assert(!structData && "fixed point type not a scalar");
    assert(scalarData.floatStandard == Float_NotFloat && "this type is a float!");
    return scalarData.bitsAmt;
  };

  int scalarIntegerBitsAmt() const
  {
    return scalarBitsAmt() - scalarFracBitsAmt();
  }

  int structSize() const
  {
    assert(structData && "fixed point type not a struct");
    return structData->size();
  }

  FixedPointType &structItem(int n)
  {
    assert(structData && "fixed point type not a struct");
    return (*structData)[n];
  }

  FixedPointType structItem(int n) const
  {
    assert(structData && "fixed point type not a struct");
    return (*structData)[n];
  }

  bool isInvalid() const
  {
    return !structData && (scalarData.bitsAmt == 0) && (scalarData.floatStandard == Float_NotFloat);
  }

  bool isFixedPoint() const
  {
    return !structData && scalarData.floatStandard == Float_NotFloat;
  }

  bool isFloatingPoint() const
  {
    return !structData && scalarData.floatStandard != Float_NotFloat;
  }

  FloatStandard getFloatingPointStandard() const
  {
    assert(scalarData.floatStandard != Float_NotFloat);
    return scalarData.floatStandard;
  }

  bool isRecursivelyInvalid() const
  {
    if (!structData)
      return isInvalid();
    for (FixedPointType &fpt : *structData) {
      if (fpt.isRecursivelyInvalid())
        return true;
    }
    return false;
  }

  FixedPointType unwrapIndexList(llvm::Type *valType, const llvm::iterator_range<const llvm::Use *> indices);

  FixedPointType unwrapIndexList(llvm::Type *valType, llvm::ArrayRef<unsigned> indices);

  bool operator==(const FixedPointType &rhs) const;
};


} // namespace flttofix


llvm::raw_ostream &operator<<(llvm::raw_ostream &stm, const flttofix::FixedPointType &f);


#undef DEBUG_TYPE

#endif
