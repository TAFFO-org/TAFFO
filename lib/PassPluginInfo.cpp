#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "Initializer/TaffoInitializer/TaffoInitializerPass.h"
#include "RangeAnalysis/TaffoVRA/ValueRangeAnalysis.hpp"

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
              }
              return false;
            });
      }};
}
