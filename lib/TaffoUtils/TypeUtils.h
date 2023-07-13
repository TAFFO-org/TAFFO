#include "InputInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"


#ifndef TAFFOUTILS_TYPEUTILS_H
#define TAFFOUTILS_TYPEUTILS_H


namespace taffo
{

/** Same as llvm::Type::isFloatingPointTy() but considers the pointer
 *  element type in case of pointers/arrays/pointers to pointers/arrays of
 *  arrays.
 *  @param scrt Source type
 *  @returns True if the pointer element type is one of the possible
 *           floating point types. */
bool isFloatType(llvm::Type *srct);

/** Finds the pointer element type of pointers to pointers and 
 *  of arrays of arrays.
 *  @param scrt Source type
 *  @returns The pointer element type. */
llvm::Type *fullyUnwrapPointerOrArrayType(llvm::Type *srct);

/** Checks if a value with the given LLVM type can have the specified InputInfo
 *  metadata attached or not.
 *  @param T An LLVM type.
 *  @param II A TAFFO InputInfo object
 *  @returns true if the two types are compatible, false otherwise. */
bool typecheckMetadata(llvm::Type *T, mdutils::MDInfo *II);

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
 *  @param outerr Pointer to a FixedPointTypeGenError which will be set
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
mdutils::FPType fixedPointTypeFromRange(
    const mdutils::Range &range,
    FixedPointTypeGenError *outerr = nullptr,
    int totalBits = 32,
    int fracThreshold = 3,
    int maxTotalBits = 64,
    int totalBitsIncrement = 64);

mdutils::PositType positTypeFromRange(
    const mdutils::Range &range,
    int minSize = 32,
    int fracThreshold = 3,
    int maxSize = 64);

} // namespace taffo


#endif // TAFFOUTILS_TYPEUTILS_H
