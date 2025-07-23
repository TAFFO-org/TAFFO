#pragma once

#include <llvm/IR/PassManager.h>

namespace taffo {

class TypeDeducerPass : public llvm::PassInfoMixin<TypeDeducerPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);

private:
  TaffoInfo& taffoInfo = TaffoInfo::getInstance();
};

} // namespace taffo
