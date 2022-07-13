#include "LLVMVersions.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"

using namespace llvm;

constexpr int BIG_NUMBER = 99999;


int taffo::getInstructionCost(
    TargetTransformInfo &TTI, Instruction *inst,
    TargetTransformInfo::TargetCostKind costKind)
{
#if (LLVM_VERSION_MAJOR >= 12)
  return TTI.getInstructionCost(inst, costKind).getValue().getValueOr(BIG_NUMBER);
#else
  return TTI.getInstructionCost(inst, costKind);
#endif
}


unsigned taffo::getStaticElementCount(const ConstantAggregateZero *V)
{
#if (LLVM_VERSION_MAJOR >= 13)
  return V->getElementCount().getFixedValue();
#else
  return V->getNumElements();
#endif
}


void taffo::CloneFunction(Function *New, const Function *Old, 
                          ValueToValueMapTy &VMap,
                          SmallVectorImpl<ReturnInst *> &Returns)
{
#if (LLVM_VERSION_MAJOR >= 13)
  CloneFunctionInto(New, Old, VMap, CloneFunctionChangeType::GlobalChanges, Returns);
#else
  CloneFunctionInto(New, Old, VMap, true, Returns);
#endif
}


unsigned taffo::numFuncArgs(const llvm::CallBase *CB)
{
#if (LLVM_VERSION_MAJOR >= 14)
  return CB->arg_size();
#else
  return CB->getNumArgOperands();
#endif
}
