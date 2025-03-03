//
// Created by nicola on 07/08/20.
//

#include "LoopAnalyzerUtil.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/Debug.h"
#include "Metadata.h"

#define DEBUG_TYPE "taffo-dta"

STATISTIC(TripCountDetectionFailCount, "Number of times the trip count of a loop could not be determined");
STATISTIC(TripCountDetectionSuccessCount, "Number of times the trip count of a loop was found");

using namespace llvm;
using namespace tuner;

unsigned tuner::computeFullTripCount(FunctionAnalysisManager& FAM, Instruction *instruction)
{
  auto bb = instruction->getParent();
  auto f = instruction->getParent()->getParent();
  auto loop = FAM.getResult<llvm::LoopAnalysis>(*f).getLoopFor(bb);

  unsigned info;
  info = computeFullTripCount(FAM, loop);

  LLVM_DEBUG(dbgs() << "Total trip count: " << info << "\n";);
  return info;
}

unsigned tuner::computeFullTripCount(FunctionAnalysisManager& FAM, Loop *loop)
{
  unsigned int LocalTrip;

  if (!loop) {
    LLVM_DEBUG(dbgs() << "Loop Info: loop is null! Not part of a loop, finishing search!\n";);
    return 1;
  }

  Function *F = (*(loop->block_begin()))->getParent();
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(*F);
  std::optional<unsigned> OUC = mdutils::MetadataManager::retrieveLoopUnrollCount(*loop, &LI);

  if (OUC.has_value()) {
    LocalTrip = OUC.value();
    if (LocalTrip > 0) {
      LLVM_DEBUG(dbgs() << "Found loop unroll count in metadata = " << LocalTrip << "\n");
      TripCountDetectionSuccessCount++;
    } else {
      LocalTrip = 2;
      LLVM_DEBUG(dbgs() << "Found loop unroll count in metadata but it's zero, forcing default of " << LocalTrip << "\n");
      TripCountDetectionFailCount++;
    }
  } else {
    LocalTrip = FAM.getResult<ScalarEvolutionAnalysis>(*loop->getHeader()->getParent()).getSmallConstantTripCount(loop);
    if (LocalTrip > 0) {
      LLVM_DEBUG(dbgs() << "SCEV told us the trip count is " << LocalTrip << ", which is OK AFAICT.\n";);
      TripCountDetectionSuccessCount++;
    } else {
      LocalTrip = 2;
      LLVM_DEBUG(dbgs() << "SCEV told us the trip count is zero; forcing the default of " << LocalTrip << "!\n");
      TripCountDetectionFailCount++;
    }
  }
  LLVM_DEBUG(dbgs() << "Checking for nested loops...\n");
  return LocalTrip * computeFullTripCount(FAM, loop->getParentLoop());
}
