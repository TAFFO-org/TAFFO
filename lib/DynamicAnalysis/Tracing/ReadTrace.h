#ifndef TAFFO_READ_TRACE
#define TAFFO_READ_TRACE

#include <unordered_map>
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

#include "TaffoUtils/InputInfo.h"

//------------------------------------------------------------------------------
// New PM interface
//------------------------------------------------------------------------------
struct ReadTrace : public llvm::PassInfoMixin<ReadTrace> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &);
  bool runOnModule(llvm::Module &M);

private:
  void
  parseTraceFiles(std::unordered_map<std::string, double>& minVals, std::unordered_map<std::string, double>& maxVals,
                  std::unordered_map<std::string, mdutils::FloatType::FloatStandard>& valTypes) const;
};

#endif
