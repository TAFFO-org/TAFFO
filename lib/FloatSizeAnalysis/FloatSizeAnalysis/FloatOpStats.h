#ifndef TAFFO_FLOATOPRANGE_H
#define TAFFO_FLOATOPRANGE_H

#include "llvm/IR/IRBuilder.h"
#include "Metadata.h"

using namespace llvm;

struct FloatOpStats {
  BinaryOperator* instruction;
  mdutils::Range op0;
  mdutils::Range op1;
  bool op0_range_set = false, op1_range_set = false;

  FloatOpStats() = default;
  FloatOpStats(const FloatOpStats &o) = default;
};

#endif // TAFFO_FLOATOPRANGE_H
