#ifndef TAFFO_NAMEVARIABLES_H
#define TAFFO_NAMEVARIABLES_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

struct NameVariables : public llvm::PassInfoMixin<NameVariables> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &);
  bool runOnModule(llvm::Module &M);
};

#endif //TAFFO_NAMEVARIABLES_H
