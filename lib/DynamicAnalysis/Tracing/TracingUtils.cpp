#include "TracingUtils.h"

namespace taffo {

using namespace llvm;

std::shared_ptr<ValueWrapper> ValueWrapper::wrapValue(llvm::Value *V)
{
  return std::make_shared<InstWrapper>(V);
}

std::shared_ptr<ValueWrapper> ValueWrapper::wrapValueUse(llvm::Use *V)
{
  std::shared_ptr<ValueWrapper> wrapper;
  if (isa<CallInst, InvokeInst>(V->getUser()) && !TracingUtils::isMallocLike(V->getUser())) {
    auto *callInst = dyn_cast<llvm::CallBase>(V->getUser());
    wrapper = std::make_shared<FunCallArgWrapper>(
        callInst, V->getOperandNo(),
        TracingUtils::isExternalCallWithPointer(callInst, V->getOperandNo())
    );
  } else {
    wrapper = std::make_shared<InstWrapper>(V->getUser());
  }
  return wrapper;
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

bool TracingUtils::isExternalCallWithPointer(const CallBase *callInst, int argNo)
{
  auto &argType = callInst->getOperandUse(argNo);
  auto *fun = callInst->getCalledFunction();
  if (!fun) {
    // conservatively consider all unknown functions with pointer arg as external
    return argType->getType()->isPointerTy();
  }
  if (argType->getType()->isPointerTy() && fun->getBasicBlockList().empty()) {
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
