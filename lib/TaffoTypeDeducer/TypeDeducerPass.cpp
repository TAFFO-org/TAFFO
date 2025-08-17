#include "Debug/Logger.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TransparentType.hpp"
#include "TypeDeducerPass.hpp"
#include "TypeDeductionAnalysis.hpp"

#define DEBUG_TYPE "taffo-typededucer"

using namespace llvm;
using namespace tda;
using namespace taffo;

PreservedAnalyses TypeDeducerPass::run(Module& m, ModuleAnalysisManager& analysisManager) {
  LLVM_DEBUG(log().logln("[TypeDeducerPass]", Logger::Magenta));
  taffoInfo.initialize(m);

  TypeDeductionAnalysis::Result& result = analysisManager.getResult<TypeDeductionAnalysis>(m);

  // Save deduced transparent types
  for (auto& [value, deducedType] : result.transparentTypes)
    if (deducedType)
      taffoInfo.setTransparentType(*value, std::move(deducedType));

  taffoInfo.dumpToFile(TYPE_DEDUCER_TAFFO_INFO, m);
  LLVM_DEBUG(log().logln("[End of TypeDeducerPass]", Logger::Magenta));
  return PreservedAnalyses::all();
}
