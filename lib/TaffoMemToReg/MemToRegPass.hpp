#pragma once

#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>

namespace taffo {

class MemToRegPass : public llvm::PassInfoMixin<MemToRegPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function& f, llvm::FunctionAnalysisManager& analysisManager);

private:
  bool promoteMemoryToRegister(llvm::Function& f,
                               llvm::DominatorTree& dominatorTree,
                               llvm::AssumptionCache& assumptionCache);
};

} // namespace taffo
