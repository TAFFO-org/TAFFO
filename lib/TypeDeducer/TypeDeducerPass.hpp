#ifndef TAFFO_TYPE_DEDUCER_PASS_HPP
#define TAFFO_TYPE_DEDUCER_PASS_HPP

#include "DeducedPointerType.hpp"
#include "InsertionOrderedMap.hpp"

#include <llvm/IR/PassManager.h>
#include <set>

namespace taffo {

class TypeDeducerPass : public llvm::PassInfoMixin<TypeDeducerPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &m, llvm::ModuleAnalysisManager &);

private:
  InsertionOrderedMap<llvm::Value*, DeducedPointerType> deducedTypes;
  llvm::DenseMap<llvm::Value*, std::set<DeducedPointerType>> candidateTypes;

  DeducedPointerType deducePointerType(llvm::Value *value);
  DeducedPointerType deduceFunctionPointerType(llvm::Function *function);
  DeducedPointerType deduceArgumentPointerType(llvm::Argument *argument);
  DeducedPointerType getDeducedType(llvm::Value *value, unsigned int additionalIndirections = 0) const;
  DeducedPointerType getBestCandidateType(const std::set<DeducedPointerType> &candidates) const;
  void logDeducedTypes() const;
};

} // namespace taffo

#endif // TAFFO_TYPE_DEDUCER_PASS_HPP
