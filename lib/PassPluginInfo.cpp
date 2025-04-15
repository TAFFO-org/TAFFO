#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include "TaffoTypeDeducer/TypeDeducerPass.hpp"
#include "TaffoInitializer/TaffoInitializer/InitializerPass.hpp"
#include "TaffoVRA/TaffoVRA/ValueRangeAnalysisPass.hpp"
#include "TaffoDTA/TaffoDTA/DataTypeAllocationPass.hpp"
#include "TaffoConversion/TaffoConversion/ConversionPass.hpp"
//#include "ErrorAnalysis/ErrorPropagator/ErrorPropagator.h"
#include "TaffoMemToReg/MemToRegPass.hpp"

using namespace llvm;
using namespace taffo;

extern "C" PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK llvmGetPassPluginInfo()
{
  return {
      LLVM_PLUGIN_API_VERSION,
      "Taffo",
      "0.4",
      [](PassBuilder &passBuilder) {
        passBuilder.registerPipelineParsingCallback(
            [](StringRef name, ModulePassManager &passManager, ArrayRef<PassBuilder::PipelineElement>) {
              if (name == "typededucer") {
                passManager.addPass(TypeDeducerPass());
                return true;
              }
              if (name == "taffoinit") {
                passManager.addPass(InitializerPass());
                return true;
              } else if (name == "taffovra") {
                passManager.addPass(ValueRangeAnalysisPass());
                return true;
              } else if (name == "taffodta") {
                passManager.addPass(tuner::DataTypeAllocationPass());
                return true;
              } else if (name == "taffoconv") {
                passManager.addPass(Conversion());
                return true;
              } /*else if (Name == "taffoerr") {
                PM.addPass(ErrorProp::ErrorPropagator());
                return true;
              }*/
              return false;
            });
        passBuilder.registerPipelineParsingCallback(
            [](StringRef name, FunctionPassManager &PM, ArrayRef<PassBuilder::PipelineElement>) {
              if (name == "taffomem2reg") {
                PM.addPass(MemToRegPass());
                return true;
              }
              return false;
            });
      }};
}
