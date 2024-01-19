#include "TracingUtils.h"

namespace taffo {

using namespace llvm;

std::shared_ptr<ValueWrapper> ValueWrapper::wrapValue(llvm::Value *V)
{
  return std::make_shared<InstWrapper>(V);
}

std::shared_ptr<ValueWrapper> ValueWrapper::wrapFunCallArg(llvm::Function *fun, unsigned int argNo)
{
  auto *formalArg = fun->getArg(argNo);
  return std::make_shared<FunCallArgWrapper>(
      formalArg, argNo,
    TracingUtils::isExternalCallWithPointer(fun, argNo)
  );
}

std::shared_ptr<ValueWrapper> ValueWrapper::wrapStructElem(llvm::Value *V, unsigned int ArgPos)
{
  return std::make_shared<StructElemWrapper>(V, ArgPos);
}

std::shared_ptr<ValueWrapper> ValueWrapper::wrapStructElemFunCallArg(llvm::Function *fun, unsigned int ArgPos, unsigned int FunArgPos)
{
  auto *formalArg = fun->getArg(FunArgPos);
  return std::make_shared<StructElemFunCallArgWrapper>(
      formalArg, ArgPos,
      FunArgPos, TracingUtils::isExternalCallWithPointer(fun, FunArgPos)
      );
}


bool TracingUtils::isMallocLike(const llvm::Function *F)
{
  const llvm::StringRef FName = F->getName();
  // TODO make sure this works in other platforms
  return FName == "malloc" || FName == "calloc" || FName == "_Znwm" || FName == "_Znam";
}

bool TracingUtils::isMallocLike(const llvm::Value *Inst)
{
  if (auto *callInst = dyn_cast<llvm::CallInst>(Inst)) {
    auto *callee = callInst->getCalledFunction();
    return callee && isMallocLike(callee);
  }
  return false;
}

bool TracingUtils::isExternalCallWithPointer(const Function *fun, unsigned int argNo)
{
  auto argType = fun->getArg(argNo)->getType();
  if (!fun) {
    // conservatively consider all unknown functions with pointer arg as external
    return argType->isPointerTy();
  }
  if (argType->isPointerTy() && fun->getBasicBlockList().empty()) {
    return !isSafeExternalFunction(fun);
  }
  return false;
}

bool TracingUtils::isSafeExternalFunction(const llvm::Function *F)
{
  const llvm::StringRef FName = F->getName();
  return FName == "free";
}

}
