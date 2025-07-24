#include "../TaffoInfo/TaffoInfo.hpp"
#include "TypeDeductionAnalysis/Debug/Logger.hpp"
#include "TypeUtils.hpp"

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "taffo"

using namespace llvm;
using namespace tda;
using namespace taffo;

Type* taffo::getFullyUnwrappedType(Value* value) {
  std::shared_ptr<tda::TransparentType> transparentType = TaffoInfo::getInstance().getOrCreateTransparentType(*value);
  return transparentType->getFullyUnwrappedType();
}

FixedPointInfo taffo::fixedPointTypeFromRange(const Range& rng,
                                              FixedPointTypeGenError* outerr,
                                              int totalBits,
                                              int fracThreshold,
                                              int maxTotalBits,
                                              int totalBitsIncrement) {
  if (outerr)
    *outerr = FixedPointTypeGenError::NoError;

  if (std::isnan(rng.min) || std::isnan(rng.max)) {
    LLVM_DEBUG(log() << "[" << __PRETTY_FUNCTION__ << "] range=" << rng.toString() << " contains NaN\n");
    if (outerr)
      *outerr = FixedPointTypeGenError::InvalidRange;
    return FixedPointInfo(true, totalBits, 0);
  }

  bool isSigned = rng.min < 0;

  if (std::isinf(rng.min) || std::isinf(rng.max)) {
    LLVM_DEBUG(log() << "[" << __PRETTY_FUNCTION__ << "] range=" << rng.toString()
                     << " contains +/-inf. Overflow may occur!\n");
    if (outerr)
      *outerr = FixedPointTypeGenError::UnboundedRange;
    return FixedPointInfo(isSigned, totalBits, 0);
  }

  double max = std::max(std::abs(rng.min), std::abs(rng.max));
  int intBit = std::lround(std::ceil(std::log2(max + 1.0))) + (isSigned ? 1 : 0);
  int bits = totalBits;

  int maxFracBitsAmt;
  if (rng.min == rng.max && fracThreshold < 0) {
    /* The range has size of zero, value is a constant.
     * Keep the value shifted as far right as possible without losing digits.
     *   TODO: This makes precision worse in the specific case where
     * the constant is at the left hand side of a division. This ideally needs
     * to be compensated by keeping track of how many significant digits we have
     * in constants.
     *   Extract exponent/mantissa from the floating point representation.
     * The exponent is effectively equal to the minimum possible size of the
     * integer part */
    int exp;
    double mant = std::frexp(max, &exp);
    /* Compute the number of non-zero bits in the mantissa */
    int nonzerobits = 0;
    while (mant != 0) {
      nonzerobits += 1;
      mant = mant * 2 - trunc(mant * 2);
    }
    /* Bound the max. number of fractional bits to the number of digits that
     * we have after the fractional dot in the original representation
     * (assumed to be the float rng.Min and rng.Max) */
    maxFracBitsAmt = std::max(0, -exp + nonzerobits);
  }
  else {
    maxFracBitsAmt = INT_MAX;
  }
  int fractionalBits = std::min(bits - intBit, maxFracBitsAmt);

  // compensate for always zero fractional bits for numbers < 0.5
  int negIntBitsAmt = std::max(0, (int) std::ceil(-std::log2(max)));

  while ((fractionalBits - negIntBitsAmt) < fracThreshold && bits < maxTotalBits) {
    bits += totalBitsIncrement;
    fractionalBits = bits - intBit;
  }

  // Check dimension
  if (fractionalBits < fracThreshold) {
    LLVM_DEBUG(log() << "[" << __PRETTY_FUNCTION__ << "] range=" << rng.toString()
                     << " Fractional part is too small!\n");
    fractionalBits = 0;
    if (intBit > bits) {
      LLVM_DEBUG(log() << "[" << __PRETTY_FUNCTION__ << "] range=" << rng.toString() << " Overflow may occur!\n");
      if (outerr)
        *outerr = FixedPointTypeGenError::NotEnoughIntAndFracBits;
    }
    else {
      if (outerr)
        *outerr = FixedPointTypeGenError::NotEnoughFracBits;
    }
  }

  return FixedPointInfo(isSigned, bits, fractionalBits);
}
