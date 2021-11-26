#include <set>
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "InstructionMix.h"


#ifndef TAFFO_ML_FEATURE_ANALYSIS_H
#define TAFFO_ML_FEATURE_ANALYSIS_H


class TaffoMLFeatureAnalysisPass : public llvm::FunctionPass {
public:
  static char ID;

  TaffoMLFeatureAnalysisPass() : llvm::FunctionPass(ID) {};

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};


#endif
