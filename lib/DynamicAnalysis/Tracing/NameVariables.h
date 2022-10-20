#ifndef TAFFO_NAMEVARIABLES_H
#define TAFFO_NAMEVARIABLES_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include <llvm/IR/InstrTypes.h>

struct NameVariables : public llvm::PassInfoMixin<NameVariables> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &);
  bool runOnModule(llvm::Module &M);

  llvm::Function* createFunctionCopy(llvm::CallBase* call);
  bool isFPFunction(llvm::Function *F);
};

#endif //TAFFO_NAMEVARIABLES_H
