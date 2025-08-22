#pragma once

#include "../TaffoInfo/ValueInfo.hpp"

#include <llvm/IR/Constants.h>

namespace taffo {

llvm::Type* getFullyUnwrappedType(llvm::Value* value);

enum class FixedPointTypeGenError {
  NoError = 0,
  InvalidRange,
  UnboundedRange,
  NotEnoughFracBits,
  NotEnoughIntAndFracBits
};

/** Generate a fixed point type appropriate for storing values
 *  contained in a given range
 *  @param range The range of values for which the type will be used
 *  @param outErr Pointer to a FixedPointTypeGenError which will be set
 *    to a value depending on the outcome of the type assignment.
 *    Optionally can be nullptr.
 *  @param totalBits The minimum amount of bits in the type
 *  @param fracThreshold The minimum amount of fractional bits in the
 *    type. If negative, the lowest amount of fractional bits that won't
 *    increase the quantization error will be chosen (at the moment,
 *    this is only relevant for zero-span ranges)
 *  @param maxTotalBits The maximum amount of bits in the type
 *  @param totalBitsIncrement The minimum amount of increment in the total
 *    amount of allocated bits to use when the range is too large for
 *    the minimum amount of bits.
 *  @returns A fixed point type. */
FixedPointInfo fixedPointInfoFromRange(const Range& range,
                                       FixedPointTypeGenError* outErr = nullptr,
                                       int totalBits = 32,
                                       int fracThreshold = 3,
                                       int maxTotalBits = 64,
                                       int totalBitsIncrement = 64);

} // namespace taffo
