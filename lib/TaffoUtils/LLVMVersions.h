#ifndef TAFFO_LLVM_VERSIONS
#define TAFFO_LLVM_VERSIONS

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Analysis/TargetTransformInfo.h"


namespace taffo {

int getInstructionCost(llvm::TargetTransformInfo &TTI, llvm::Instruction *inst,
                       llvm::TargetTransformInfo::TargetCostKind costKind);

unsigned getStaticElementCount(const llvm::ConstantAggregateZero *V);

void CloneFunction(llvm::Function *New, const llvm::Function *Old,
                   llvm::ValueToValueMapTy &VMap,
                   llvm::SmallVectorImpl<llvm::ReturnInst *> &Returns);

unsigned numFuncArgs(const llvm::CallBase *CB);

}


#endif // TAFFO_LLVM_VERSIONS
