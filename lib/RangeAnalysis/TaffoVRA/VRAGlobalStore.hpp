#ifndef TAFFO_VRA_GLOBAL_STORE_HPP
#define TAFFO_VRA_GLOBAL_STORE_HPP

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Module.h"

#include "CodeInterpreter.hpp"
#include "VRALogger.hpp"
#include "VRAStore.hpp"

namespace taffo
{

class VRAGlobalStore : protected VRAStore, public AnalysisStore
{
public:
  VRAGlobalStore()
      : VRAStore(VRASK_VRAGlobalStore, std::make_shared<VRALogger>()),
        AnalysisStore(ASK_VRAGlobalStore) {}

  void convexMerge(const AnalysisStore &Other) override;
  std::shared_ptr<CodeAnalyzer> newCodeAnalyzer(CodeInterpreter &CI) override;
  std::shared_ptr<AnalysisStore> newFunctionStore(CodeInterpreter &CI) override;

  bool hasValue(const llvm::Value *V) const override
  {
    auto It = DerivedRanges.find(V);
    return (It != DerivedRanges.end() && It->second) || (V && llvm::isa<llvm::Constant>(V));
  }

  std::shared_ptr<CILogger> getLogger() const override { return Logger; }

  // Metadata Processing
  void harvestMetadata(llvm::Module &M);
  NodePtrT harvestStructMD(const mdutils::MDInfo *MD, const llvm::Type *T);
  void saveResults(llvm::Module &M);
  bool isValidRange(const mdutils::Range *rng) const;
  void refreshRange(const llvm::Instruction *i);
  std::shared_ptr<mdutils::MDInfo> toMDInfo(const RangeNodePtrT r);
  void updateMDInfo(std::shared_ptr<mdutils::MDInfo> mdi, const RangeNodePtrT r);
  static void setConstRangeMetadata(mdutils::MetadataManager &MDManager,
                                    llvm::Instruction &i);

  const range_ptr_t fetchRange(const llvm::Value *V) override;
  using VRAStore::fetchRange;
  const RangeNodePtrT fetchRangeNode(const llvm::Value *V) override;
  NodePtrT getNode(const llvm::Value *v) override;
  void setNode(const llvm::Value *V, NodePtrT Node) override
  {
    VRAStore::setNode(V, Node);
  }
  RangeNodePtrT getUserInput(const llvm::Value *V) const;
  NodePtrT fetchConstant(const llvm::Constant *v);

  static bool classof(const AnalysisStore *AS)
  {
    return AS->getKind() == ASK_VRAGlobalStore;
  }

  static bool classof(const VRAStore *VS)
  {
    return VS->getKind() == VRASK_VRAGlobalStore;
  }

protected:
  llvm::DenseMap<const llvm::Value *, RangeNodePtrT> UserInput;
};

} // end namespace taffo

#endif
