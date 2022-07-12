//===-- FixedPoint.h - Representation of fixed-point values -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of
/// fixed point arithmetic type wrappers.
///
//===----------------------------------------------------------------------===//

#ifndef ERRORPROPAGATOR_FIXED_POINT_H
#define ERRORPROPAGATOR_FIXED_POINT_H

#include <cstdint>

#include "AffineForms.h"
#include "InputInfo.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"

#define DEBUG_TYPE "errorprop"

namespace ErrorProp
{

/// Intermediate type for error computations.
typedef long double inter_t;

/// Interval of former fixed point values
/// An interval representing a fixed point range in the intermediate type.
class FPInterval : public Interval<inter_t>
{
public:
  FPInterval() : IInfo(nullptr) {}

  FPInterval(const mdutils::InputInfo *II) : IInfo(II)
  {
    assert(II != nullptr);
    assert(II->IRange != nullptr);

    this->Min = getMin();
    this->Max = getMax();
  }

  FPInterval(const Interval<inter_t> &I)
      : Interval<inter_t>(I), IInfo(nullptr) {}

  bool hasInitialError() const
  {
    return IInfo != nullptr && IInfo->IError != nullptr;
  }

  double getInitialError() const
  {
    if (hasInitialError())
      return *IInfo->IError;
    else
      return 0.0;
  }

  inter_t getRoundingError() const
  {
    if (!isUninitialized() && IInfo->IType != nullptr)
      return static_cast<inter_t>(IInfo->IType->getRoundingError());
    else
      return 0.0;
  }

  bool isUninitialized() const { return IInfo == nullptr; }

  const mdutils::TType *getTType() const
  {
    if (isUninitialized())
      return nullptr;

    return IInfo->IType.get();
  }

protected:
  const mdutils::InputInfo *IInfo;

  inter_t getMin() const
  {
    return static_cast<inter_t>(IInfo->IRange->Min);
  }

  inter_t getMax() const
  {
    return static_cast<inter_t>(IInfo->IRange->Max);
  }
};

/// Fixed Point value type wrapper.
/// Represents the type of a fixed point value
/// as the total number of bits of the implementation (precision)
/// and the number of fractional bits (PointPos).
class FixedPointValue
{
public:
  virtual unsigned getPrecision() const = 0;

  virtual unsigned getPointPos() const
  {
    return PointPos;
  }

  virtual bool isSigned() const = 0;

  virtual FPInterval getInterval() const = 0;

  virtual llvm::MDNode *toMetadata(llvm::LLVMContext &) const = 0;

  virtual ~FixedPointValue() = default;

  static std::unique_ptr<FixedPointValue>
  createFromConstantInt(int SPrec,
                        const llvm::IntegerType *IT,
                        const llvm::ConstantInt *CIMin,
                        const llvm::ConstantInt *CIMax);

  static std::unique_ptr<FixedPointValue>
  createFromMDNode(const llvm::IntegerType *IT, const llvm::MDNode &N);

  static std::unique_ptr<FixedPointValue>
  createFromMetadata(const llvm::Instruction &);

protected:
  const unsigned PointPos; ///< Number of fractional binary digits.

  FixedPointValue(const unsigned PointPos)
      : PointPos(PointPos) {}

  llvm::ConstantAsMetadata *
  getPointPosMetadata(llvm::LLVMContext &C, bool Signed = false) const;
};

/// This class represents an interval of 32 bit unsigned fixed point values.
class UFixedPoint32 : public FixedPointValue
{
public:
  UFixedPoint32(const unsigned PointPos);

  UFixedPoint32(const unsigned PointPos,
                const uint32_t Min, const uint32_t Max);

  unsigned getPrecision() const override
  {
    return 32U;
  }

  bool isSigned() const override
  {
    return false;
  }

  FPInterval getInterval() const override;

  llvm::MDNode *toMetadata(llvm::LLVMContext &) const override;

protected:
  uint32_t Min;
  uint32_t Max;
};

/// This class represents an interval of 64 bit unsigned fixed point values.
class UFixedPoint64 : public FixedPointValue
{
public:
  UFixedPoint64(const unsigned PointPos);

  UFixedPoint64(const unsigned PointPos,
                const uint64_t Min, const uint64_t Max);

  unsigned getPrecision() const override
  {
    return 64U;
  }

  bool isSigned() const override
  {
    return false;
  }

  FPInterval getInterval() const override;

  llvm::MDNode *toMetadata(llvm::LLVMContext &) const override;

protected:
  uint64_t Min;
  uint64_t Max;
};

class SFixedPoint32 : public FixedPointValue
{
public:
  SFixedPoint32(const unsigned PointPos);

  SFixedPoint32(const unsigned PointPos,
                const int32_t Min, const int32_t Max);

  unsigned getPrecision() const override
  {
    return 32U;
  }

  bool isSigned() const override
  {
    return true;
  }

  FPInterval getInterval() const override;

  llvm::MDNode *toMetadata(llvm::LLVMContext &) const override;

protected:
  int32_t Min;
  int32_t Max;
};

class SFixedPoint64 : public FixedPointValue
{
public:
  SFixedPoint64(const unsigned PointPos);

  SFixedPoint64(const unsigned PointPos,
                const int64_t Min, const int64_t Max);

  unsigned getPrecision() const override
  {
    return 64U;
  }

  bool isSigned() const override
  {
    return true;
  }

  FPInterval getInterval() const override;

  llvm::MDNode *toMetadata(llvm::LLVMContext &) const override;

protected:
  int64_t Min;
  int64_t Max;
};

class FixedPointGeneric : public FixedPointValue
{
public:
  FixedPointGeneric(const unsigned PointPos, const unsigned Precision, const bool Signed);
  FixedPointGeneric(const unsigned PointPos, const unsigned Precision, const bool Signed,
                    const llvm::APInt &Min, const llvm::APInt &Max);

  unsigned getPrecision() const override
  {
    return Precision;
  }

  bool isSigned() const override
  {
    return Signed;
  }

  FPInterval getInterval() const override;

  llvm::MDNode *toMetadata(llvm::LLVMContext &) const override;

protected:
  llvm::APInt Min;
  llvm::APInt Max;
  unsigned Precision;
  bool Signed;
};

} // end namespace ErrorProp

#undef DEBUG_TYPE // "errorprop"

#endif // ERRORPROPAGATOR_FIXED_POINT_H
