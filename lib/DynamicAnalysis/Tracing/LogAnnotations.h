#ifndef TAFFO_LOGANNOTATIONS_H
#define TAFFO_LOGANNOTATIONS_H


#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include <llvm/IR/InstrTypes.h>

struct LogAnnotations : public llvm::PassInfoMixin<LogAnnotations> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &);
  bool runOnModule(llvm::Module &M);
};

#endif // TAFFO_LOGANNOTATIONS_H
