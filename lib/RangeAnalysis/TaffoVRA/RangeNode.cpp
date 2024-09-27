#include "RangeNode.hpp"
#include "TypeUtils.h"
#include <llvm/Support/Debug.h>

using namespace llvm;

StructType *taffo::VRAStructNode::getStructType() const
{
  auto UT = taffo::fullyUnwrapPointerOrArrayType(Type);
  assert(UT->isStructTy());
  return cast<StructType>(UT);
}
