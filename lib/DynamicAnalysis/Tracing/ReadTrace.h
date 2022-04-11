#ifndef TAFFO_READ_TRACE
#define TAFFO_READ_TRACE

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

//------------------------------------------------------------------------------
// New PM interface
//------------------------------------------------------------------------------
struct ReadTrace : public llvm::PassInfoMixin<ReadTrace> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &);
  bool runOnModule(llvm::Module &M);
};

#endif
