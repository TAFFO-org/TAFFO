#ifndef TAFFO_READ_TRACE
#define TAFFO_READ_TRACE

#include <unordered_map>
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

#include "TaffoUtils/InputInfo.h"
#include "MemoryGraph.h"

extern llvm::cl::opt<bool> Fixm;
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

  std::unordered_map<int, std::list<std::shared_ptr<taffo::ValueWrapper>>> ccValues;
  std::unordered_map<int, std::pair<double, double>> ccRanges;
  std::unordered_map<llvm::Value*, llvm::SmallVector<mdutils::InputInfo *>> constInfo;
  std::unordered_map<llvm::Value*, std::shared_ptr<mdutils::InputInfo>> valuesInfo;
  std::unordered_map<llvm::Type*, std::shared_ptr<mdutils::StructInfo>> structsInfo;
  std::unordered_map<llvm::Function*, std::shared_ptr<llvm::SmallVector<std::shared_ptr<mdutils::MDInfo>>>> functionsInfo;

  std::string typeName(const llvm::Value& val);

  void calculateCCRanges(const std::unordered_map<int, std::list<std::shared_ptr<taffo::ValueWrapper>>>& ccValues,
                                    const std::unordered_map<std::string, double>& minVals,
                                    const std::unordered_map<std::string, double>& maxVals,
                                    std::unordered_map<int, std::pair<double, double>>& ccRanges);

  void
  parseTraceFiles(std::unordered_map<std::string, double>& minVals, std::unordered_map<std::string, double>& maxVals,
                  std::unordered_map<std::string, mdutils::FloatType::FloatStandard>& valTypes) const;

  std::shared_ptr<mdutils::StructInfo> addStructInfo(
      std::shared_ptr<taffo::ValueWrapper> valueWrapper,
      const std::pair<double, double> &range,
      bool disableConversion);

  void setAllMetadata(llvm::Module &M);
};

#endif
