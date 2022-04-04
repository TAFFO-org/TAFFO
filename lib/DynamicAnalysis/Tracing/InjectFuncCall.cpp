//========================================================================
// FILE:
//    InjectFuncCall.cpp
//
// DESCRIPTION:
//    For each function defined in the input IR module, InjectFuncCall inserts
//    a call to printf (from the C standard I/O library). The injected IR code
//    corresponds to the following function call in ANSI C:
//    ```C
//      printf("(llvm-tutor) Hello from: %s\n(llvm-tutor)   number of arguments: %d\n",
//             FuncName, FuncNumArgs);
//    ```
//    This code is inserted at the beginning of each function, i.e. before any
//    other instruction is executed.
//
//    To illustrate, for `void foo(int a, int b, int c)`, the code added by InjectFuncCall
//    will generated the following output at runtime:
//    ```
//    (llvm-tutor) Hello World from: foo
//    (llvm-tutor)   number of arguments: 3
//    ```
//
// USAGE:
//    1. Legacy pass manager:
//      $ opt -load <BUILD_DIR>/lib/libInjectFuncCall.so `\`
//        --legacy-inject-func-call <bitcode-file>
//    2. New pass maanger:
//      $ opt -load-pass-plugin <BUILD_DIR>/lib/libInjectFunctCall.so `\`
//        -passes=-"inject-func-call" <bitcode-file>
//
// License: MIT
//========================================================================
#include "InjectFuncCall.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;

#define DEBUG_TYPE "inject-func-call"

//-----------------------------------------------------------------------------
// InjectFuncCall implementation
//-----------------------------------------------------------------------------
bool InjectFuncCall::runOnModule(Module &M) {
  bool InsertedAtLeastOnePrintf = false;

  auto &CTX = M.getContext();
  PointerType *PrintfArgTy = PointerType::getUnqual(Type::getInt8Ty(CTX));

  // STEP 1: Inject the declaration of printf
  // ----------------------------------------
  // Create (or _get_ in cases where it's already available) the following
  // declaration in the IR module:
  //    declare i32 @printf(i8*, ...)
  // It corresponds to the following C declaration:
  //    int printf(char *, ...)
  FunctionType *PrintfTy = FunctionType::get(
      IntegerType::getInt32Ty(CTX),
      PrintfArgTy,
      /*IsVarArgs=*/true);

  FunctionCallee Printf = M.getOrInsertFunction("printf", PrintfTy);

  // Set attributes as per inferLibFuncAttributes in BuildLibCalls.cpp
  Function *PrintfF = dyn_cast<Function>(Printf.getCallee());
  PrintfF->setDoesNotThrow();
  PrintfF->addParamAttr(0, Attribute::NoCapture);
  PrintfF->addParamAttr(0, Attribute::ReadOnly);


  // STEP 2: Inject a global variable that will hold the printf format string
  // ------------------------------------------------------------------------
  llvm::Constant *PrintfFormatStr = llvm::ConstantDataArray::getString(CTX, "%s: %.17g\n");

  Constant *PrintfFormatStrVar =
      M.getOrInsertGlobal("PrintfFormatStr", PrintfFormatStr->getType());
  dyn_cast<GlobalVariable>(PrintfFormatStrVar)->setInitializer(PrintfFormatStr);

  // STEP 3: For each function in the module, inject a call to printf
  // ----------------------------------------------------------------
  IRBuilder<> Builder(CTX);
  long counter = 0;
  auto moduleName = M.getModuleIdentifier();

  auto getVarName = [&moduleName](long counter) -> std::string {
    return (Twine(moduleName) + Twine("::var") + Twine(counter)).str();
  };

  for (auto &F : M) {
    if (!F.hasName() || F.isDeclaration())
      continue;

    for (auto &BB: F.getBasicBlockList()) {
      auto &InstList = BB.getInstList();
      auto current = InstList.getNextNode(InstList.front());
      while (current != nullptr) {
        auto &Inst = *current;
        auto next = InstList.getNextNode(*current);
        if (!Inst.isDebugOrPseudoInst() && Inst.getType()->isFloatingPointTy()) {
          Inst.setName(getVarName(counter));
          counter++;
          if (next != nullptr) {
            Builder.SetInsertPoint(next);
          } else {
            Builder.SetInsertPoint(&BB.back());
          }
          auto FuncName = Builder.CreateGlobalStringPtr(Inst.getName());
          // Printf requires i8*, but PrintfFormatStrVar is an array: [n x i8]. Add a cast: [n x i8] -> i8*
          llvm::Value *FormatStrPtr = Builder.CreatePointerCast(
              PrintfFormatStrVar, PrintfArgTy, "formatStr");
          Builder.CreateCall(Printf, {FormatStrPtr, FuncName, &Inst});
          InsertedAtLeastOnePrintf = true;
        }
        current = next;
      }
    }
  }

  return InsertedAtLeastOnePrintf;
}

PreservedAnalyses InjectFuncCall::run(llvm::Module &M,
                                       llvm::ModuleAnalysisManager &) {
  bool Changed =  runOnModule(M);

  return (Changed ? llvm::PreservedAnalyses::none()
                  : llvm::PreservedAnalyses::all());
}

bool LegacyInjectFuncCall::runOnModule(llvm::Module &M) {
  bool Changed = Impl.runOnModule(M);

  return Changed;
}

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getInjectFuncCallPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "inject-func-call", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "inject-func-call") {
                    MPM.addPass(InjectFuncCall());
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

//-----------------------------------------------------------------------------
// Legacy PM Registration
//-----------------------------------------------------------------------------
char LegacyInjectFuncCall::ID = 0;

// Register the pass - required for (among others) opt
static RegisterPass<LegacyInjectFuncCall>
    X(/*PassArg=*/"legacy-inject-func-call", /*Name=*/"LegacyInjectFuncCall",
      /*CFGOnly=*/false, /*is_analysis=*/false);
