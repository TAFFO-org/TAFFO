//
// Created by nicola on 07/08/20.
//

#include "LoopAnalyzerUtil.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "taffo-dta"

using namespace llvm;

unsigned LoopAnalyzerUtil::computeFullTripCount(ModulePass *tuner, Instruction *instruction)
{
  auto bb = instruction->getParent();
  auto f = instruction->getParent()->getParent();
  auto loop = tuner->getAnalysis<LoopInfoWrapperPass>(*f).getLoopInfo().getLoopFor(bb);

  unsigned info;
  info = computeFullTripCount(tuner, loop);

  LLVM_DEBUG(dbgs() << "Total trip count: " << info << "\n";);
  return info;
}

unsigned LoopAnalyzerUtil::computeFullTripCount(ModulePass *tuner, Loop *loop)
{
  if (!loop) {
    LLVM_DEBUG(dbgs() << "Loop Info: loop is null! Not part of a loop, finishing search!\n";);
    return 1;
  }

  auto scev = tuner->getAnalysis<ScalarEvolutionWrapperPass>(
                       *loop->getHeader()->getParent())
                  .getSE()
                  .getSmallConstantTripCount(loop);
  if (scev == 0) {
    scev = 2;
    LLVM_DEBUG(dbgs() << "SCEV told us the trip count is zero; forcing the default of " << scev << "!\n");
  } else {
    LLVM_DEBUG(dbgs() << "SCEV told us the trip count is " << scev << ", which is OK AFAICT.\n";);
  }
  LLVM_DEBUG(dbgs() << "Checking for nested loops...\n");
  return scev * computeFullTripCount(tuner, loop->getParentLoop());
}
