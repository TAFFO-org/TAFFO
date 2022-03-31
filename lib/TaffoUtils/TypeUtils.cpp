#include "TypeUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "taffo"


using namespace taffo;
using namespace llvm;


Type *taffo::fullyUnwrapPointerOrArrayType(Type *srct)
{
  if (srct->isPointerTy()) {
    return fullyUnwrapPointerOrArrayType(srct->getPointerElementType());
  } else if (srct->isArrayTy()) {
    return fullyUnwrapPointerOrArrayType(srct->getArrayElementType());
  }
  return srct;
}


bool taffo::isFloatType(Type *srct)
{
  return fullyUnwrapPointerOrArrayType(srct)->isFloatingPointTy();
}


mdutils::FPType taffo::fixedPointTypeFromRange(
    const mdutils::Range &rng,
    FixedPointTypeGenError *outerr,
    int totalBits,
    int fracThreshold,
    int maxTotalBits,
    int totalBitsIncrement)
{
  if (outerr)
    *outerr = FixedPointTypeGenError::NoError;

  if (std::isnan(rng.Min) || std::isnan(rng.Max)) {
    LLVM_DEBUG(dbgs() << "[" << __PRETTY_FUNCTION__ << "] range=" << rng.toString() << " contains NaN\n");
    if (outerr)
      *outerr = FixedPointTypeGenError::InvalidRange;
    return mdutils::FPType(totalBits, 0, true);
  }

  bool isSigned = rng.Min < 0;

  if (std::isinf(rng.Min) || std::isinf(rng.Max)) {
    LLVM_DEBUG(dbgs() << "[" << __PRETTY_FUNCTION__ << "] range=" << rng.toString() << " contains +/-inf. Overflow may occur!\n");
    if (outerr)
      *outerr = FixedPointTypeGenError::UnboundedRange;
    return mdutils::FPType(totalBits, 0, isSigned);
  }

  double max = std::max(std::abs(rng.Min), std::abs(rng.Max));
  int intBit = std::lround(std::ceil(std::log2(static_cast<int>(max) + 1.0))) + (isSigned ? 1 : 0);
  int bitsAmt = totalBits;

  int maxFracBitsAmt;
  if (rng.Min == rng.Max && fracThreshold < 0) {
    int exp;
    double mant = std::frexp(max, &exp);
    // rng.Min == rng.Max == mant * (2 ** exp)
    int nonzerobits = 0;
    while (mant != 0) {
      nonzerobits += 1;
      mant = mant * 2 - trunc(mant * 2);
    }
    maxFracBitsAmt = std::max(0, -exp + nonzerobits);
  } else {
    maxFracBitsAmt = INT_MAX;
  }
  int fracBitsAmt = std::min(bitsAmt - intBit, maxFracBitsAmt);

  // compensate for always zero fractional bits for numbers < 0.5
  int negIntBitsAmt = std::max(0, (int)std::ceil(-std::log2(max)));

  while ((fracBitsAmt - negIntBitsAmt) < fracThreshold && bitsAmt < maxTotalBits) {
    bitsAmt += totalBitsIncrement;
    fracBitsAmt = bitsAmt - intBit;
  }

  // Check dimension
  if (fracBitsAmt < fracThreshold) {
    LLVM_DEBUG(dbgs() << "[" << __PRETTY_FUNCTION__ << "] range=" << rng.toString() << " Fractional part is too small!\n");
    fracBitsAmt = 0;
    if (intBit > bitsAmt) {
      LLVM_DEBUG(dbgs() << "[" << __PRETTY_FUNCTION__ << "] range=" << rng.toString() << " Overflow may occur!\n");
      if (outerr)
        *outerr = FixedPointTypeGenError::NotEnoughIntAndFracBits;
    } else {
      if (outerr)
        *outerr = FixedPointTypeGenError::NotEnoughFracBits;
    }
  }

  return mdutils::FPType(bitsAmt, fracBitsAmt, isSigned);
}
