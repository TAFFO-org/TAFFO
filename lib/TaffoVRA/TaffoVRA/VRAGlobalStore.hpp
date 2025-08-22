#pragma once

#include "CodeInterpreter.hpp"
#include "VRALogger.hpp"
#include "VRAStore.hpp"

#include <llvm/ADT/DenseMap.h>

#define DEBUG_TYPE "taffo-vra"

namespace taffo {

class VRAGlobalStore : protected VRAStore,
                       public AnalysisStore {
public:
  VRAGlobalStore()
  : VRAStore(VRASK_VRAGlobalStore, std::make_shared<VRALogger>()), AnalysisStore(ASK_VRAGlobalStore) {}

  void convexMerge(const AnalysisStore& other) override;
  std::shared_ptr<CodeAnalyzer> newCodeAnalyzer(CodeInterpreter& CI) override;
  std::shared_ptr<AnalysisStore> newFunctionStore(CodeInterpreter& CI) override;

  bool hasValue(const llvm::Value* V) const override {
    auto It = DerivedRanges.find(V);
    return (It != DerivedRanges.end() && It->second) || (V && llvm::isa<llvm::Constant>(V));
  }

  std::shared_ptr<CILogger> getLogger() const override { return Logger; }

  // Metadata Processing
  void harvestValueInfo(llvm::Module& m);
  void saveResults(llvm::Module& m);
  bool isValidRange(const Range* rng) const;
  void updateValueInfo(const std::shared_ptr<ValueInfo>& valueInfo,
                       const std::shared_ptr<ValueInfoWithRange>& valueInfoWithRange);
  static void setConstRangeMetadata(llvm::Instruction& inst);

  std::shared_ptr<Range> fetchRange(const llvm::Value* v) override;
  using VRAStore::fetchRange;
  std::shared_ptr<ValueInfoWithRange> fetchRangeNode(const llvm::Value* v) override;
  std::shared_ptr<ValueInfo> getNode(const llvm::Value* v) override;
  void setNode(const llvm::Value* V, const std::shared_ptr<ValueInfo> Node) override { VRAStore::setNode(V, Node); }
  std::shared_ptr<ValueInfoWithRange> getUserInput(const llvm::Value* v) const;
  std::shared_ptr<ValueInfo> fetchConstant(const llvm::Constant* constant);

  static bool classof(const AnalysisStore* AS) { return AS->getKind() == ASK_VRAGlobalStore; }

  static bool classof(const VRAStore* VS) { return VS->getKind() == VRASK_VRAGlobalStore; }

protected:
  llvm::DenseMap<const llvm::Value*, std::shared_ptr<ValueInfoWithRange>> UserInput;
};

} // end namespace taffo

#undef DEBUG_TYPE
