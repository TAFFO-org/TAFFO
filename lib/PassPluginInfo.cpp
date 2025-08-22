#include "TaffoConversion/ConversionPass.hpp"
#include "TaffoDTA/DataTypeAllocationPass.hpp"
#include "TaffoInitializer/InitializerPass.hpp"
#include "TaffoMemToReg/MemToRegPass.hpp"
#include "TaffoTypeDeducer/TypeDeducerPass.hpp"
#include "TaffoVRA/TaffoVRA/ValueRangeAnalysisPass.hpp"
#include "TypeDeductionAnalysis.hpp"

#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

using namespace llvm;
using namespace tda;
using namespace taffo;

extern "C" PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Taffo", "1.0", [](PassBuilder& passBuilder) {
            passBuilder.registerPipelineParsingCallback(
              [](StringRef name, ModulePassManager& passManager, ArrayRef<PassBuilder::PipelineElement>) {
                if (name == "typededucer") {
                  passManager.addPass(TypeDeducerPass());
                  return true;
                }
                if (name == "taffoinit") {
                  passManager.addPass(InitializerPass());
                  return true;
                }
                if (name == "taffovra") {
                  passManager.addPass(ValueRangeAnalysisPass());
                  return true;
                }
                if (name == "taffodta") {
                  passManager.addPass(taffo::DataTypeAllocationPass());
                  return true;
                }
                if (name == "taffoconv") {
                  passManager.addPass(ConversionPass());
                  return true;
                }
                return false;
              });
            passBuilder.registerPipelineParsingCallback(
              [](StringRef name, FunctionPassManager& passManager, ArrayRef<PassBuilder::PipelineElement>) {
                if (name == "taffomem2reg") {
                  passManager.addPass(MemToRegPass());
                  return true;
                }
                return false;
              });
            passBuilder.registerAnalysisRegistrationCallback([](ModuleAnalysisManager& moduleAnalysisManager) {
              moduleAnalysisManager.registerPass([] { return TypeDeductionAnalysis(); });
            });
          }};
}
