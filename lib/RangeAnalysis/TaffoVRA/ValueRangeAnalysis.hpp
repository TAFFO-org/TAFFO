#ifndef TAFFO_VALUE_RANGE_ANALYSIS_HPP
#define TAFFO_VALUE_RANGE_ANALYSIS_HPP

#include "CodeInterpreter.hpp"

#include <llvm/IR/PassManager.h>
#include <llvm/Support/CommandLine.h>

#define DEBUG_TYPE "taffo-vra"

namespace taffo
{

extern llvm::cl::opt<bool> PropagateAll;
extern llvm::cl::opt<unsigned> Unroll;
extern llvm::cl::opt<unsigned> MaxUnroll;

class ValueRangeAnalysis : public llvm::PassInfoMixin<ValueRangeAnalysis> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

private:
  void processModule(CodeInterpreter &CodeInt, llvm::Module &M);
};

} // namespace taffo

#undef DEBUG_TYPE

#endif
