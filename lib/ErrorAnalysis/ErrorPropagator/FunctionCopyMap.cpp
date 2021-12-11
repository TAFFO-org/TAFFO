#include "FunctionCopyMap.h"
#include "Metadata.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/Config/llvm-config.h"

namespace ErrorProp
{

using namespace llvm;

#define DEBUG_TYPE "errorprop"

void UnrollLoops(Pass &P, Function &F, unsigned DefaultUnrollCount, unsigned MaxUnroll)
{
  // Prepare required analyses
  LoopInfo &LInfo = P.getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
  SmallVector<Loop *, 4U> Loops(LInfo.begin(), LInfo.end());

  // Now try to unroll all loops
  for (Loop *L : Loops) {
    ScalarEvolution &SE = P.getAnalysis<ScalarEvolutionWrapperPass>(F).getSE();
    // Compute loop trip count
    unsigned TripCount = SE.getSmallConstantTripCount(L);
    // Get user supplied unroll count
    Optional<unsigned> OUC = mdutils::MetadataManager::retrieveLoopUnrollCount(*L, &LInfo);
    unsigned UnrollCount = DefaultUnrollCount;
    if (OUC.hasValue())
      if (TripCount != 0 && OUC.getValue() > TripCount)
        UnrollCount = TripCount;
      else
        UnrollCount = OUC.getValue();
    else if (TripCount != 0)
      UnrollCount = TripCount;

    if (UnrollCount > MaxUnroll)
      UnrollCount = MaxUnroll;

    LLVM_DEBUG(dbgs() << "Trying to unroll loop by " << UnrollCount << "... ");

    unsigned TripMult = SE.getSmallConstantTripMultiple(L);
    if (TripMult == 0U)
      TripMult = UnrollCount;

    // Actually unroll loop
    DominatorTree &DomTree = P.getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
    AssumptionCache &AssC = P.getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    OptimizationRemarkEmitter &ORE = P.getAnalysis<OptimizationRemarkEmitterWrapperPass>(F).getORE();
    TargetTransformInfo &TTI = P.getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    UnrollLoopOptions ULO;
    #if (LLVM_VERSION_MAJOR >= 13)
    ULO = {
      .Count = UnrollCount,
      .Force = true,
      .Runtime = false,
      .AllowExpensiveTripCount = true,
      .UnrollRemainder = false,
      .ForgetAllSCEV = false};
    #else
    ULO = {
      .Count = UnrollCount,
      .TripCount = TripCount,
      .Force = true,
      .AllowRuntime = false,
      .AllowExpensiveTripCount = true,
      .PreserveCondBr = false,
      .PreserveOnlyFirst = false,
      .TripMultiple = TripMult,
      .PeelCount = 0U,
      .UnrollRemainder = false,
      .ForgetAllSCEV = false};
    #endif

    LoopUnrollResult URes = UnrollLoop(L, ULO, &LInfo, &SE, &DomTree, &AssC, &TTI, &ORE, false);

    switch (URes) {
    case LoopUnrollResult::Unmodified:
      LLVM_DEBUG(dbgs() << "unmodified.\n");
      break;
    case LoopUnrollResult::PartiallyUnrolled:
      LLVM_DEBUG(dbgs() << "unrolled partially.\n");
      break;
    case LoopUnrollResult::FullyUnrolled:
      LLVM_DEBUG(dbgs() << "done.\n");
      break;
    }
  }
}

FunctionCopyCount *FunctionCopyManager::prepareFunctionData(Function *F)
{
  assert(F != nullptr);

  auto FCData = FCMap.find(F);
  if (FCData == FCMap.end()) {
    // If no copy of F has already been made, create one, so loop transformations
    // do not change original code.
    FunctionCopyCount &FCC = FCMap[F];

    if ((FCC.MaxRecCount = mdutils::MetadataManager::retrieveMaxRecursionCount(*F)) == 0U)
      FCC.MaxRecCount = MaxRecursionCount;

    // Check if we really need to clone the function
    if (MaxUnroll > 0U && !F->empty()) {
      LoopInfo &LInfo =
          P.getAnalysis<LoopInfoWrapperPass>(*F).getLoopInfo();
      if (!LInfo.empty()) {
        FCC.Copy = CloneFunction(F, FCC.VMap);

        if (FCC.Copy != nullptr)
          UnrollLoops(P, *FCC.Copy, DefaultUnrollCount, MaxUnroll);
      }
    }
    return &FCC;
  }
  return &FCData->second;
}

FunctionCopyManager::~FunctionCopyManager()
{
  for (auto &FCC : FCMap) {
    if (FCC.second.Copy != nullptr)
      FCC.second.Copy->eraseFromParent();
  }
}

} // end namespace ErrorProp
