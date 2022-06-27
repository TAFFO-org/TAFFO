#ifndef TAFFO_STRIPANNOTATIONS_H
#define TAFFO_STRIPANNOTATIONS_H


#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include <llvm/IR/InstrTypes.h>

struct StripAnnotations : public llvm::PassInfoMixin<StripAnnotations> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &);
  bool runOnModule(llvm::Module &M);
};

#endif // TAFFO_STRIPANNOTATIONS_H
