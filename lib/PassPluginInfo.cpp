#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "Initializer/TaffoInitializer/TaffoInitializerPass.h"
#include "RangeAnalysis/TaffoVRA/ValueRangeAnalysis.hpp"
#include "DataTypeAlloc/TaffoDTA/TaffoDTA.h"
#include "Conversion/LLVMFloatToFixed/LLVMFloatToFixedPass.h"
#include "ErrorAnalysis/ErrorPropagator/ErrorPropagator.h"
#include "TaffoMem2Reg/Mem2Reg.h"

using namespace llvm;
using namespace taffo;

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK llvmGetPassPluginInfo()
{
  return {
      LLVM_PLUGIN_API_VERSION,
      "Taffo",
      "0.3",
      [](PassBuilder &PB) {
        PB.registerPipelineParsingCallback(
            [](StringRef Name, ModulePassManager &PM, ArrayRef<PassBuilder::PipelineElement>) {
              if (Name == "taffoinit") {
                PM.addPass(TaffoInitializer());
                return true;
              } else if (Name == "taffovra") {
                PM.addPass(ValueRangeAnalysis());
                return true;
              } else if (Name == "taffodta") {
                PM.addPass(tuner::TaffoTuner());
                return true;
              } else if (Name == "taffoconv") {
                PM.addPass(flttofix::Conversion());
                return true;
              } else if (Name == "taffoerr") {
                PM.addPass(ErrorProp::ErrorPropagator());
                return true;
              }
              return false;
            });
        PB.registerPipelineParsingCallback(
            [](StringRef Name, FunctionPassManager &PM, ArrayRef<PassBuilder::PipelineElement>) {
              if (Name == "taffomem2reg") {
                PM.addPass(taffo::TaffoMem2Reg());
                return true;
              }
              return false;
            });
      }};
}
