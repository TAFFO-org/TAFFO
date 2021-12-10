#include "InputInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/User.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <fstream>


#ifndef __FIXED_POINT_TYPE_H__
#define __FIXED_POINT_TYPE_H__


namespace flttofix
{


class FixedPointType
{
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

  FixedPointType(mdutils::TType *mdtype);

  static FixedPointType get(mdutils::MDInfo *mdnfo, int *enableConversion = nullptr);

  std::string toString() const;

  llvm::Type *scalarToLLVMType(llvm::LLVMContext &ctxt) const;

  inline bool &scalarIsSigned(void)
  {
    assert(!structData && "fixed point type not a scalar");
    return scalarData.isSigned;
  };

  inline bool scalarIsSigned(void) const
  {
    assert(!structData && "fixed point type not a scalar");
    return scalarData.isSigned;
  };

  inline int &scalarFracBitsAmt(void)
  {
    assert(!structData && "fixed point type not a scalar");
    assert(scalarData.floatStandard == Float_NotFloat && "this type is a float!");
    return scalarData.fracBitsAmt;
  };

  inline int scalarFracBitsAmt(void) const
  {
    assert(!structData && "fixed point type not a scalar");
    assert(scalarData.floatStandard == Float_NotFloat && "this type is a float!");
    return scalarData.fracBitsAmt;
  };

  inline int &scalarBitsAmt(void)
  {
    assert(!structData && "fixed point type not a scalar");
    assert(scalarData.floatStandard == Float_NotFloat && "this type is a float!");
    return scalarData.bitsAmt;
  };

  inline int scalarBitsAmt(void) const
  {
    assert(!structData && "fixed point type not a scalar");
    assert(scalarData.floatStandard == Float_NotFloat && "this type is a float!");
    return scalarData.bitsAmt;
  };

  inline int structSize(void) const
  {
    assert(structData && "fixed point type not a struct");
    return structData->size();
  }

  inline FixedPointType &structItem(int n)
  {
    assert(structData && "fixed point type not a struct");
    return (*structData)[n];
  }

  inline FixedPointType structItem(int n) const
  {
    assert(structData && "fixed point type not a struct");
    return (*structData)[n];
  }

  inline bool isInvalid(void) const
  {
    return !structData && (scalarData.bitsAmt == 0) && (scalarData.floatStandard == Float_NotFloat);
  }

  inline bool isFixedPoint() const
  {
    return !structData && scalarData.floatStandard == Float_NotFloat;
  }

  inline bool isFloatingPoint() const
  {
    return !structData && scalarData.floatStandard != Float_NotFloat;
  }

  inline FloatStandard getFloatingPointStandard() const
  {
    assert(scalarData.floatStandard != Float_NotFloat);
    return scalarData.floatStandard;
  }

  inline bool isRecursivelyInvalid(void) const
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


#endif
