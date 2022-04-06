#ifndef DYNAMIC_TRACING
#define DYNAMIC_TRACING

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/IR/Type.h"

//------------------------------------------------------------------------------
// New PM interface
//------------------------------------------------------------------------------
struct InjectFuncCall : public llvm::PassInfoMixin<InjectFuncCall> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &);
  bool runOnModule(llvm::Module &M);
};

#endif
