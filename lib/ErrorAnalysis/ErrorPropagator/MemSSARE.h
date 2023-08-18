#ifndef ERRORPROPAGATOR_MEMSSARE_H
#define ERRORPROPAGATOR_MEMSSARE_H

#include "llvm/ADT/SmallVector.h"
#include "MemSSAUtils.hpp"
#include "RangeErrorMap.h"

#define DEBUG_TYPE "errorprop"

namespace ErrorProp
{

class MemSSARE : private taffo::MemSSAUtils
{
public:
  typedef llvm::SmallVector<const RangeErrorMap::RangeError *, DEFAULT_RANGE_COUNT> REVector;

  MemSSARE(RangeErrorMap &RMap, llvm::MemorySSA &MemSSA)
      : MemSSAUtils(MemSSA), RMap(RMap) {}

  REVector &getRangeErrors(llvm::Instruction *I, bool Sloppy=false);

private:
  RangeErrorMap &RMap;
  REVector Res;
  void findLOEError(llvm::Instruction *I);
};

} // end of namespace ErrorProp

#undef DEBUG_TYPE // "errorprop"

#endif
