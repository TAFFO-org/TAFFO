#include "Debug/Logger.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "VRAGlobalStore.hpp"
#include "ValueRangeAnalysisPass.hpp"

#include <llvm/Analysis/MemorySSA.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "taffo-vra"

using namespace llvm;
using namespace tda;
using namespace taffo;

namespace taffo {

cl::opt<bool> PropagateAll("propagate-all",
                           cl::desc("Propagate ranges for all functions, not only those marked as starting point."),
                           cl::init(false));

cl::opt<unsigned> Unroll("unroll",
                         cl::desc("Default loop unroll count. Setting this to 0 disables loop unrolling. (Default: 1)"),
                         cl::value_desc("count"),
                         cl::init(1U));

cl::opt<unsigned>
  MaxUnroll("max-unroll",
            cl::desc("Max loop unroll count. Setting this to 0 disables loop unrolling. (Default: 256)"),
            cl::value_desc("count"),
            cl::init(256U));

} // namespace taffo

PreservedAnalyses ValueRangeAnalysisPass::run(Module& M, ModuleAnalysisManager& AM) {
  LLVM_DEBUG(log().logln("[ValueRangeAnalysisPass]", Logger::Magenta));

  // No need to initialize if memToReg is run before in the same opt call
  // TaffoInfo::getInstance().initializeFromFile("taffo_info_memToReg.json", M);

  std::shared_ptr<VRAGlobalStore> GlobalStore = std::make_shared<VRAGlobalStore>();
  GlobalStore->harvestValueInfo(M);

  CodeInterpreter CodeInt(AM, GlobalStore, Unroll, MaxUnroll);
  processModule(CodeInt, M);

  LLVM_DEBUG(log() << "saving results...\n");
  GlobalStore->saveResults(M);

  TaffoInfo::getInstance().dumpToFile("taffo_info_vra.json", M);
  LLVM_DEBUG(log().logln("[End of ValueRangeAnalysisPass]", Logger::Magenta));
  return PreservedAnalyses::all();
}

void ValueRangeAnalysisPass::processModule(CodeInterpreter& CodeInt, Module& M) {
  bool FoundVisitableFunction = false;
  for (Function& F : M) {
    if (!F.empty() && (PropagateAll || TaffoInfo::getInstance().isStartingPoint(F))) {
      CodeInt.interpretFunction(&F);
      FoundVisitableFunction = true;
    }
  }

  if (!FoundVisitableFunction)
    LLVM_DEBUG(log() << " No visitable functions found.\n");
}
