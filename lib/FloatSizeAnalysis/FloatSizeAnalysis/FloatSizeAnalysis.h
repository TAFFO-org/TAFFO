#ifndef TAFFO_FLOATSIZEANALYSIS_H
#define TAFFO_FLOATSIZEANALYSIS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include <llvm/IR/InstrTypes.h>

struct FloatSizeAnalysis : public llvm::PassInfoMixin<FloatSizeAnalysis> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &);
  bool runOnModule(llvm::Module &M);
  void printOpRanges(llvm::BinaryOperator *binOp);
};


#endif // TAFFO_FLOATSIZEANALYSIS_H
