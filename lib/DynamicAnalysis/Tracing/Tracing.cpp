#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "InjectFuncCall.h"
#include "ReadTrace.h"
#include "NameVariables.h"

using namespace llvm;

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getInjectFuncCallPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "dynamic-tracing", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                    [](StringRef Name, ModulePassManager &MPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                      if (Name == "inject-func-call") {
                        MPM.addPass(InjectFuncCall());
                        return true;
                      }
                      if (Name == "read-trace") {
                        MPM.addPass(ReadTrace());
                        return true;
                      }
                      if (Name == "name-variables") {
                        MPM.addPass(NameVariables());
                        return true;
                      }
                      return false;
                    });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getInjectFuncCallPluginInfo();
}
