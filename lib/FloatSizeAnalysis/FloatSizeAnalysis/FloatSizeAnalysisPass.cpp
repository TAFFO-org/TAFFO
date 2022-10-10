#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "FloatSizeAnalysis.h"


using namespace llvm;

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getInjectFuncCallPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "float-size-analysis", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "float-size-analysis") {
                    MPM.addPass(FloatSizeAnalysis());
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

//------------------------------------------------------------------------------
// Legacy PM interface
//------------------------------------------------------------------------------
struct LegacyFloatSizeAnalysis : public llvm::ModulePass {
  static char ID;
  LegacyFloatSizeAnalysis() : ModulePass(ID) {}
  bool runOnModule(llvm::Module &M) override {
    bool Changed = Impl.runOnModule(M);
    return Changed;
  }

  FloatSizeAnalysis Impl;
};

char LegacyFloatSizeAnalysis::ID = 0;

// Register the pass - required for (among others) opt
static RegisterPass<LegacyFloatSizeAnalysis>
    X(/*PassArg=*/"taffo-float-size-analysis", /*Name=*/"TAFFO Framework analyze float size",
      /*CFGOnly=*/false, /*is_analysis=*/true);
