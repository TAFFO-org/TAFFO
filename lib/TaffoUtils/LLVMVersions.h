#include "llvm/Analysis/TargetTransformInfo.h"


int getInstructionCost(llvm::TargetTransformInfo& TTI, llvm::Instruction * inst, llvm::TargetTransformInfo::TargetCostKind costKind );
