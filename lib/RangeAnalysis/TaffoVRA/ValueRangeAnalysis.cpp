#include "ValueRangeAnalysis.hpp"

#include "VRAGlobalStore.hpp"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "taffo-vra"

using namespace llvm;
using namespace taffo;
using namespace mdutils;

namespace taffo
{
llvm::cl::opt<bool> PropagateAll("propagate-all",
                                 llvm::cl::desc("Propagate ranges for all functions, "
                                                "not only those marked as starting point."),
                                 llvm::cl::init(false));
llvm::cl::opt<unsigned> Unroll("unroll",
                               llvm::cl::desc("Default loop unroll count. "
                                              "Setting this to 0 disables loop unrolling. "
                                              "(Default: 1)"),
                               llvm::cl::value_desc("count"),
                               llvm::cl::init(1U));

llvm::cl::opt<unsigned> MaxUnroll("max-unroll",
                                  llvm::cl::desc("Max loop unroll count. "
                                                 "Setting this to 0 disables loop unrolling. "
                                                 "(Default: 256)"),
                                  llvm::cl::value_desc("count"),
                                  llvm::cl::init(256U));
}

PreservedAnalyses ValueRangeAnalysis::run(Module &M, ModuleAnalysisManager &AM)
{
  std::shared_ptr<VRAGlobalStore> GlobalStore = std::make_shared<VRAGlobalStore>();
  GlobalStore->harvestMetadata(M);

  CodeInterpreter CodeInt(AM, GlobalStore, Unroll, MaxUnroll);
  processModule(CodeInt, M);

  LLVM_DEBUG(dbgs() << "saving results...\n");
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
