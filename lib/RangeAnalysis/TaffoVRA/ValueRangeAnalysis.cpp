#include "ValueRangeAnalysis.hpp"

#include "llvm/Support/Debug.h"
#include "llvm/Analysis/MemorySSA.h"
#include "VRAGlobalStore.hpp"

using namespace llvm;
using namespace taffo;
using namespace mdutils;

char ValueRangeAnalysis::ID = 0;

static RegisterPass<ValueRangeAnalysis> X(
	"taffoVRA",
	"TAFFO Framework Value Range Analysis Pass",
	false /* does not affect the CFG */,
	false /* only Analysis */);

bool
ValueRangeAnalysis::runOnModule(Module &M) {
  std::shared_ptr<VRAGlobalStore> GlobalStore = std::make_shared<VRAGlobalStore>();
  GlobalStore->harvestMetadata(M);

  CodeInterpreter CodeInt(*this, GlobalStore, Unroll, MaxUnroll);
  processModule(CodeInt, M);

  GlobalStore->saveResults(M);

  return true;
}

void
ValueRangeAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  CodeInterpreter::getAnalysisUsage(AU);
  AU.addRequiredTransitive<MemorySSAWrapperPass>();
  AU.setPreservesAll();
}

void
ValueRangeAnalysis::processModule(CodeInterpreter &CodeInt, Module &M) {
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
