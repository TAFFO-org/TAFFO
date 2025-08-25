#include "../ConversionPass.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include <map>

using namespace llvm;
using namespace tda;
using namespace taffo;

#define DEBUG_TYPE "taffo-conv"

void ConversionPass::convertIndirectCalls() {
  using IndirectCallHandler = void (ConversionPass::*)(CallBase* trampolineCall, Function* indirectFun);
  const std::map<const std::string, IndirectCallHandler> indirectCallFunctions = {
    {"__kmpc_fork_call", &ConversionPass::handleKmpcFork}
  };

  // Convert the trampoline calls
  for (auto&& [trampolineCall, indirectFun] : taffoInfo.getIndirectFunctions()) {
    auto iter = indirectCallFunctions.find(static_cast<std::string>(indirectFun->getName()));
    if (iter != indirectCallFunctions.end())
      (this->*iter->second)(trampolineCall, indirectFun);
  }
}

void ConversionPass::handleKmpcFork(CallBase* trampolineCall, Function* indirectFunction) {
  auto* convertedCall = cast<CallBase>(convertedValues.at(trampolineCall));

  // Get converted microtask
  Function* convertedTrampolineFun = convertedCall->getCalledFunction();
  CallBase* convertedMicrotaskCall = nullptr;
  for (auto& inst : instructions(convertedTrampolineFun))
    if (auto* call = dyn_cast<CallBase>(&inst)) {
      convertedMicrotaskCall = call;
      break;
    }
  assert(convertedMicrotaskCall && "Could not find call to converted microtask");
  Function* convertedMicrotask = convertedMicrotaskCall->getCalledFunction();

  // Add the indirect arguments
  std::vector<Value*> indirectCallArgs = std::vector<Value*>(convertedCall->arg_begin(), convertedCall->arg_end());
  indirectCallArgs.insert(indirectCallArgs.begin() + 2, convertedMicrotask);

  // Insert the indirect call after the patched direct call
  CallInst* indirectCall = CallInst::Create(indirectFunction, indirectCallArgs);
  indirectCall->insertAfter(trampolineCall);
  copyValueInfo(indirectCall, trampolineCall);

  taffoInfo.eraseValue(convertedCall);
}
