#include "../TaffoInfo/TaffoInfo.hpp"
#include "Debug/Logger.hpp"
#include "TypeUtils.hpp"

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "taffo-common"

using namespace llvm;
using namespace tda;
using namespace taffo;

Type* taffo::getFullyUnwrappedType(Value* value) {
  TransparentType* transparentType = TaffoInfo::getInstance().getOrCreateTransparentType(*value);
  return transparentType->getFullyUnwrappedType()->toLLVMType();
}

FixedPointInfo taffo::fixedPointInfoFromRange(const Range& range,
                                              FixedPointTypeGenError* outErr,
                                              int totalBits,
                                              int fracThreshold,
                                              int maxTotalBits,
                                              int totalBitsIncrement) {
  if (outErr)
    *outErr = FixedPointTypeGenError::NoError;

  if (std::isnan(range.min) || std::isnan(range.max)) {
    LLVM_DEBUG(log() << "[" << __PRETTY_FUNCTION__ << "] range=" << range.toString() << " contains NaN\n");
    if (outErr)
      *outErr = FixedPointTypeGenError::InvalidRange;
    return FixedPointInfo(true, totalBits, 0);
  }

  bool isSigned = range.min < 0;

  if (std::isinf(range.min) || std::isinf(range.max)) {
    LLVM_DEBUG(log() << "[" << __PRETTY_FUNCTION__ << "] range=" << range.toString()
                     << " contains +/-inf. Overflow may occur!\n");
    if (outErr)
      *outErr = FixedPointTypeGenError::UnboundedRange;
    return FixedPointInfo(isSigned, totalBits, 0);
  }

  double max = std::max(std::abs(range.min), std::abs(range.max));
  int intBit = std::lround(std::ceil(std::log2(max + 1.0))) + (isSigned ? 1 : 0);
  int bits = totalBits;

  int maxFracBitsAmt;
  if (range.min == range.max && fracThreshold < 0) {
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
    bits = std::min(bits, maxTotalBits);
    fractionalBits = bits - intBit;
  }

  // Check dimension
  if (fractionalBits < fracThreshold) {
    LLVM_DEBUG(log() << "[" << __PRETTY_FUNCTION__ << "] range=" << range.toString()
                     << " Fractional part is too small!\n");
    fractionalBits = 0;
    if (intBit > bits) {
      LLVM_DEBUG(log() << "[" << __PRETTY_FUNCTION__ << "] range=" << range.toString() << " Overflow may occur!\n");
      if (outErr)
        *outErr = FixedPointTypeGenError::NotEnoughIntAndFracBits;
    }
    else {
      if (outErr)
        *outErr = FixedPointTypeGenError::NotEnoughFracBits;
    }
  }

  return FixedPointInfo(isSigned, bits, fractionalBits);
}
