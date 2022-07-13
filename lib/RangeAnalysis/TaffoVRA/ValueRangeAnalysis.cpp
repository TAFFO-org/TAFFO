#include "ValueRangeAnalysis.hpp"

#include "VRAGlobalStore.hpp"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "taffo-vra"

using namespace llvm;
using namespace taffo;
using namespace mdutils;

cl::opt<bool> taffo::PropagateAll("propagate-all",
                                  cl::desc("Propagate ranges for all functions, "
                                           "not only those marked as starting point."),
                                  cl::init(false));
cl::opt<unsigned> taffo::Unroll("unroll",
                                cl::desc("Default loop unroll count. "
                                         "Setting this to 0 disables loop unrolling. "
                                         "(Default: 1)"),
                                cl::value_desc("count"),
                                cl::init(1U));

cl::opt<unsigned> taffo::MaxUnroll("max-unroll",
                                   cl::desc("Max loop unroll count. "
                                            "Setting this to 0 disables loop unrolling. "
                                            "(Default: 256)"),
                                   cl::value_desc("count"),
                                   cl::init(256U));

PreservedAnalyses ValueRangeAnalysis::run(Module &M, ModuleAnalysisManager &AM)
{
  std::shared_ptr<VRAGlobalStore> GlobalStore = std::make_shared<VRAGlobalStore>();
  GlobalStore->harvestMetadata(M);

  CodeInterpreter CodeInt(AM, GlobalStore, Unroll, MaxUnroll);
  processModule(CodeInt, M);

  GlobalStore->saveResults(M);

  return PreservedAnalyses::all();
}

void ValueRangeAnalysis::processModule(CodeInterpreter &CodeInt, Module &M)
{
  bool FoundVisitableFunction = false;
  for (llvm::Function &F : M) {
    if (!F.empty() && (PropagateAll || MetadataManager::isStartingPoint(F))) {
      CodeInt.interpretFunction(&F);
      FoundVisitableFunction = true;
    }
  }

  if (!FoundVisitableFunction) {
    LLVM_DEBUG(dbgs() << DEBUG_HEAD << " No visitable functions found.\n");
  }
}
