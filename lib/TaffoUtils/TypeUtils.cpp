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


bool taffo::areEquivalentType(Type *T1, Type *T2)
{
  while (T1->isPointerTy() && T2->isPointerTy()) {
    T1 = T1->getPointerElementType();
    T2 = T2->getPointerElementType();
  }
  if (!(T1->isStructTy() && T2->isStructTy()))
    return T1 == T2;
  auto structT1 = cast<StructType>(T1);
  auto structT2 = cast<StructType>(T2);
  if ((structT1->hasName() && structT2->hasName()) ||
      (!structT1->hasName() && !structT2->hasName()))
    return T1 == T2;
  if (structT1->getNumElements() != structT2->getNumElements())
    return false;
  for (unsigned int i = 0; i < structT1->getNumElements(); i++) {
    if (!areEquivalentType(structT1->getElementType(i), structT2->getElementType(i)))
      return false;
  }
  return true;
}


bool taffo::typecheckMetadata(mdutils::MDInfo *II, llvm::Type *T, const DataLayout &DL)
{
  assert(T);
  assert(II);
  LLVM_DEBUG(dbgs() << "[typecheckMetadata] Typechecking " << II->toString() << " against LLVM type " << *T << "\n");
  llvm::Type *UT = fullyUnwrapPointerOrArrayType(T);

  if (UT->isSingleValueType() && isa<mdutils::InputInfo>(II)) {
    LLVM_DEBUG(dbgs() << "[typecheckMetadata] Check OK\n");
    II->setType(UT, DL);
    return true;
  }

  if (UT->isStructTy() && isa<mdutils::StructInfo>(II)) {
    auto *ST = dyn_cast<StructType>(UT);
    auto *SI = dyn_cast<mdutils::StructInfo>(II);
    LLVM_DEBUG(dbgs() << "Struct type: " << *ST << "\n");
    if (UT->getStructNumElements() != SI->size()) {
      LLVM_DEBUG(dbgs() << "[typecheckMetadata] Check failed: struct member count mismatch (expected "
                        << UT->getStructNumElements() << ", got " << SI->size() << ")\n");
      return false;
    }
    for (unsigned i = 0; i < SI->size(); i++) {
      mdutils::MDInfo *IIField = SI->getField(i).get();
      auto *TyField = ST->elements()[i];
      if (IIField != nullptr) {
        if (!typecheckMetadata(IIField, TyField, DL)) {
          LLVM_DEBUG(dbgs() << "[typecheckMetadata] Check failed for one of the struct fields\n");
          return false;
        }
      }
    }
    LLVM_DEBUG(dbgs() << "[typecheckMetadata] Check OK\n");
    II->setType(T, DL);
    return true;
  }
  
  LLVM_DEBUG(dbgs() << "[typecheckMetadata] Check failed: struct/single value mismatch\n");
  return false;
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
  int intBit = std::lround(std::ceil(std::log2(max + 1.0))) + (isSigned ? 1 : 0);
  int bitsAmt = totalBits;

  int maxFracBitsAmt;
  if (rng.Min == rng.Max && fracThreshold < 0) {
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
