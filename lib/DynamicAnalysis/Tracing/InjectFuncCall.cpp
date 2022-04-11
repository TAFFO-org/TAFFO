#include "InjectFuncCall.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Compiler.h"

#include <unordered_map>

#include "TaffoUtils/InputInfo.h"

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
  llvm::Constant *PrintfFormatStr = llvm::ConstantDataArray::getString(CTX, "\nTAFFO_TRACE %s %f %s\n");

  Constant *PrintfFormatStrVar =
      M.getOrInsertGlobal("PrintfFormatStr", PrintfFormatStr->getType());
  dyn_cast<GlobalVariable>(PrintfFormatStrVar)->setInitializer(PrintfFormatStr);

  // STEP 3: For each function in the module, inject a call to printf
  // ----------------------------------------------------------------
  IRBuilder<> Builder(CTX);
  std::unordered_map<Type::TypeID, Constant*> floatTypeNameConstants;

  for (auto type: mdutils::FloatType::llvmFloatTypes) {
    auto typeName = mdutils::FloatType::getFloatStandardName(type);
    floatTypeNameConstants[type] = Builder.CreateGlobalStringPtr(typeName, "", 0, &M);
  }

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
          if (next != nullptr) {
            Builder.SetInsertPoint(next);
          } else {
            Builder.SetInsertPoint(&BB.back());
          }
          auto InstName = Builder.CreateGlobalStringPtr(Inst.getName());
          // Printf requires i8*, but PrintfFormatStrVar is an array: [n x i8]. Add a cast: [n x i8] -> i8*
          llvm::Value *FormatStrPtr = Builder.CreatePointerCast(PrintfFormatStrVar, PrintfArgTy, "formatStr");

          Builder.CreateCall(Printf, {
            FormatStrPtr,
            InstName,
            &Inst,
            floatTypeNameConstants[Inst.getType()->getTypeID()]
          });
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
