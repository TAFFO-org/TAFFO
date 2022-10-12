#ifndef TAFFO_FLOATSIZEANALYSIS_H
#define TAFFO_FLOATSIZEANALYSIS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include <llvm/IR/InstrTypes.h>
#include "FloatOpStats.h"

struct FloatSizeAnalysis : public llvm::PassInfoMixin<FloatSizeAnalysis> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &);
  bool runOnModule(llvm::Module &M);
  void getOpRanges(llvm::BinaryOperator *binOp);
  std::unique_ptr<mdutils::Range> rangeFromValue(Value *op);

  std::list<FloatOpStats> stats;
  void printStatsCSV();
  int minExponent(mdutils::Range &range);
  int maxExponent(mdutils::Range &range);
  int maxExponentDiff(mdutils::Range &range1, mdutils::Range &range2);
};


#endif // TAFFO_FLOATSIZEANALYSIS_H
