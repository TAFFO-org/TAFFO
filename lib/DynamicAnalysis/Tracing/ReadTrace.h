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

  int buildMemEdgesList(llvm::Module &M, std::unordered_map<llvm::Value*, int>& instToIndex,
                         std::unordered_map<int, llvm::Value*>& indexToInst,
                         std::list<std::pair<int, int>>& edges);

  void connectedComponents(int count, std::list<std::pair<int, int>>& edges, std::map<int, std::list<int>>& cc);

  void
  parseTraceFiles(std::unordered_map<std::string, double>& minVals, std::unordered_map<std::string, double>& maxVals,
                  std::unordered_map<std::string, mdutils::FloatType::FloatStandard>& valTypes) const;
};

#endif
