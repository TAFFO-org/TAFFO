#include "LLVMVersions.h"


constexpr int BIG_NUMBER = 99999;


int getInstructionCost(llvm::TargetTransformInfo &TTI, llvm::Instruction *inst, llvm::TargetTransformInfo::TargetCostKind costKind)
{

#if (LLVM_VERSION_MAJOR == 12)
  return TTI.getInstructionCost(inst, costKind).getValue().getValueOr(BIG_NUMBER);
#endif
#if (LLVM_VERSION_MAJOR == 11)
  return TTI.getInstructionCost(inst, costKind);
#endif
}
