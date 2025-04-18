#pragma once

#include "Containers/InsertionOrderedMap.hpp"
#include "Types/TransparentType.hpp"

#include <llvm/IR/PassManager.h>

#include <unordered_set>

namespace taffo {

class TypeDeducerPass : public llvm::PassInfoMixin<TypeDeducerPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);

private:
  using CandidateSet = std::unordered_set<std::shared_ptr<TransparentType>>;

  InsertionOrderedMap<llvm::Value*, std::shared_ptr<TransparentType>> deducedTypes;
  llvm::DenseMap<llvm::Value*, CandidateSet> candidateTypes;
  TaffoInfo& taffoInfo = TaffoInfo::getInstance();

  std::shared_ptr<TransparentType> deducePointerType(llvm::Value* value);
  std::shared_ptr<TransparentType> deduceFunctionPointerType(llvm::Function* function);
  std::shared_ptr<TransparentType> deduceArgumentPointerType(llvm::Argument* argument);
  std::shared_ptr<TransparentType> getDeducedType(llvm::Value* value) const;
  std::shared_ptr<TransparentType> getBestCandidateType(const CandidateSet& candidates) const;

  void logDeduction(llvm::Value* value,
                    const std::shared_ptr<TransparentType>& bestCandidate,
                    const CandidateSet& candidates);
  void logDeducedTypes();
};

} // namespace taffo
