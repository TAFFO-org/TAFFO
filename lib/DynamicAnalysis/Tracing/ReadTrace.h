#ifndef TAFFO_READ_TRACE
#define TAFFO_READ_TRACE

#include <unordered_map>
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

#include "TaffoUtils/InputInfo.h"
#include "MemoryGraph.h"

//------------------------------------------------------------------------------
// New PM interface
//------------------------------------------------------------------------------
struct ReadTrace : public llvm::PassInfoMixin<ReadTrace> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &);
  bool runOnModule(llvm::Module &M);

private:

  struct DynamicValueInfo {
    double min;
    double max;
    bool disableConversion = false;

    DynamicValueInfo(double Min, double Max, bool DisableConversion)
        : min(Min), max(Max), disableConversion(DisableConversion) {}
  };

  std::string typeName(const llvm::Value& val);

  void calculateCCRanges(const std::unordered_map<int, std::list<std::shared_ptr<taffo::ValueWrapper>>>& ccValues,
                                    const std::unordered_map<std::string, double>& minVals,
                                    const std::unordered_map<std::string, double>& maxVals,
                                    std::unordered_map<int, std::pair<double, double>>& ccRanges);

  int buildMemEdgesList(llvm::Module &M, std::unordered_map<llvm::Value*, int>& instToIndex,
                         std::unordered_map<int, llvm::Value*>& indexToInst,
                         std::list<std::pair<int, int>>& edges);

  void connectedComponents(const int count, const std::list<std::pair<int, int>>& edges,
                           std::unordered_map<int, std::list<int>>& cc);

  void
  parseTraceFiles(std::unordered_map<std::string, double>& minVals, std::unordered_map<std::string, double>& maxVals,
                  std::unordered_map<std::string, mdutils::FloatType::FloatStandard>& valTypes) const;
  bool disableConversionForExternalFun(const llvm::Value* v);
};

#endif
