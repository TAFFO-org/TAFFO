#pragma once

#include "CodeInterpreter.hpp"

#include <llvm/IR/PassManager.h>
#include <llvm/Support/CommandLine.h>

#define DEBUG_TYPE "taffo-vra"

namespace taffo
{

extern llvm::cl::opt<bool> PropagateAll;
extern llvm::cl::opt<unsigned> Unroll;
extern llvm::cl::opt<unsigned> MaxUnroll;

class ValueRangeAnalysisPass : public llvm::PassInfoMixin<ValueRangeAnalysisPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

private:
  void processModule(CodeInterpreter &CodeInt, llvm::Module &M);
};

} // namespace taffo

#undef DEBUG_TYPE
