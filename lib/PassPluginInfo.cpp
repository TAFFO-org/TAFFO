#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "Initializer/TaffoInitializer/TaffoInitializerPass.h"
#include "RangeAnalysis/TaffoVRA/ValueRangeAnalysis.hpp"
#include "DataTypeAlloc/TaffoDTA/TaffoDTA.h"
#include "Conversion/LLVMFloatToFixed/LLVMFloatToFixedPass.h"
#include "ErrorAnalysis/ErrorPropagator/ErrorPropagator.h"
#include "TaffoMem2Reg/Mem2Reg.h"
#include "DynamicAnalysis/Tracing/InjectFuncCall.h"
#include "DynamicAnalysis/Tracing/ReadTrace.h"
#include "DynamicAnalysis/Tracing/NameVariables.h"
#include "DynamicAnalysis/Tracing/StripAnnotations.h"
#include "FloatSizeAnalysis/FloatSizeAnalysis/FloatSizeAnalysis.h"


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
              if (Name == "inject-func-call") {
                PM.addPass(InjectFuncCall());
                return true;
              }
              if (Name == "read-trace") {
                PM.addPass(ReadTrace());
                return true;
              }
              if (Name == "name-variables") {
                PM.addPass(NameVariables());
                return true;
              }
              if (Name == "strip-annotations") {
                PM.addPass(StripAnnotations());
                return true;
              }
              if (Name == "float-size-analysis") {
                PM.addPass(FloatSizeAnalysis());
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

//------------------------------------------------------------------------------
// Legacy PM interface
//------------------------------------------------------------------------------
struct LegacyInjectFuncCall : public llvm::ModulePass {
  static char ID;
  LegacyInjectFuncCall() : ModulePass(ID) {}
  bool runOnModule(llvm::Module &M) override {
    bool Changed = Impl.runOnModule(M);
    return Changed;
  }

  InjectFuncCall Impl;
};

struct LegacyNameVariables : public llvm::ModulePass {
  static char ID;
  LegacyNameVariables() : ModulePass(ID) {}
  bool runOnModule(llvm::Module &M) override {
    bool Changed = Impl.runOnModule(M);
    return Changed;
  }

  NameVariables Impl;
};

struct LegacyReadTrace : public llvm::ModulePass {
  static char ID;
  LegacyReadTrace() : ModulePass(ID) {}
  bool runOnModule(llvm::Module &M) override {
    bool Changed = Impl.runOnModule(M);
    return Changed;
  }

  ReadTrace Impl;
};

struct LegacyStripAnnotations : public llvm::ModulePass {
  static char ID;
  LegacyStripAnnotations() : ModulePass(ID) {}
  bool runOnModule(llvm::Module &M) override {
    bool Changed = Impl.runOnModule(M);
    return Changed;
  }

  StripAnnotations Impl;
};

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
char LegacyInjectFuncCall::ID = 0;
char LegacyNameVariables::ID = 0;
char LegacyReadTrace::ID = 0;
char LegacyStripAnnotations::ID = 0;

// Register the pass - required for (among others) opt
static RegisterPass<LegacyInjectFuncCall>
    X(/*PassArg=*/"taffo-inject-func-call", /*Name=*/"TAFFO Framework inject tracing",
      /*CFGOnly=*/false, /*is_analysis=*/false);

static RegisterPass<LegacyNameVariables>
    Y(/*PassArg=*/"taffo-name-variables", /*Name=*/"TAFFO Framework assign unique names to registers",
      /*CFGOnly=*/false, /*is_analysis=*/false);

static RegisterPass<LegacyReadTrace>
    Z(/*PassArg=*/"taffo-read-trace", /*Name=*/"TAFFO Framework read range information from trace files",
      /*CFGOnly=*/false, /*is_analysis=*/false);

static RegisterPass<LegacyStripAnnotations>
    U(/*PassArg=*/"taffo-strip-annotations", /*Name=*/"TAFFO Framework strip annotations from file",
      /*CFGOnly=*/false, /*is_analysis=*/false);

static RegisterPass<LegacyFloatSizeAnalysis>
    F(/*PassArg=*/"taffo-float-size-analysis", /*Name=*/"TAFFO Framework analyze float size",
      /*CFGOnly=*/false, /*is_analysis=*/true);

