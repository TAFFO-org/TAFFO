#include "Debug/Logger.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TypeDeducerPass.hpp"
#include "TypeDeductionAnalysis.hpp"
#include "TypeDispatcher.hpp"

#include <utility>

#define DEBUG_TYPE "taffo-typededucer"

using namespace llvm;
using namespace tda;
using namespace taffo;

PreservedAnalyses TypeDeducerPass::run(Module& m, ModuleAnalysisManager& analysisManager) {
  LLVM_DEBUG(log().logln("[TypeDeducerPass]", Logger::Magenta));
  dispatcher.registerModule(m);
  taffoInfo.initialize(m);

  TypeDeductionAnalysis::Result& result = analysisManager.getResult<TypeDeductionAnalysis>(m);

  // Save deduced transparent types
  taffoInfo.initializeFromResult(std::move(result));

  taffoInfo.dumpToFile(TYPE_DEDUCER_TAFFO_INFO, m);
  dispatcher.unregisterModule(m);
  LLVM_DEBUG(log().logln("[End of TypeDeducerPass]", Logger::Magenta));
  return PreservedAnalyses::all();
}
