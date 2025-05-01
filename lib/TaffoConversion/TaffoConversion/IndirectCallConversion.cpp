#include "ConversionPass.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include <map>

using namespace llvm;
using namespace taffo;
using namespace taffo;

#define DEBUG_TYPE "taffo-conversion"

/// Retrieve the indirect calls converted into trampolines and re-use the
/// original indirect functions.
void FloatToFixed::convertIndirectCalls(Module& m) {
  using handler_function = void (FloatToFixed::*)(CallInst* patchedDirectCall, Function* indirectFunction);
  const std::map<const std::string, handler_function> indirectCallFunctions = {
    {"__kmpc_fork_call", &FloatToFixed::handleKmpcFork}
  };

  std::vector<CallInst*> trampolineCalls;

  // Retrieve the trampoline calls using the INDIRECT_METADATA
  for (Function& curFunction : m) {
    for (auto instructionIt = inst_begin(curFunction); instructionIt != inst_end(curFunction); instructionIt++) {
      if (auto curCallInstruction = dyn_cast<CallInst>(&(*instructionIt))) {
        if (TaffoInfo::getInstance().isIndirectFunction(*curCallInstruction))
          trampolineCalls.push_back(curCallInstruction);
      }
    }
  }

  // Convert the trampoline calls
  for (auto trampolineCall : trampolineCalls) {
    auto* indirectFunction = TaffoInfo::getInstance().getIndirectFunction(*trampolineCall);

    if (indirectFunction == nullptr) {
      LLVM_DEBUG(log() << "Blocking the following conversion for failed "
                           "dyn_cast on the indirect function: "
                        << *trampolineCall << "\n");
      continue;
    }

    auto indirectCallHandler = indirectCallFunctions.find((std::string) indirectFunction->getName());

    if (indirectCallHandler != indirectCallFunctions.end()) {
      handler_function indirectFunctionHandler = indirectCallHandler->second;
      (this->*indirectFunctionHandler)(trampolineCall, indirectFunction);
    }
  }
}

/// Convert a trampoline call to an outlined function back into the original
/// library function
void FloatToFixed::handleKmpcFork(CallInst* patchedDirectCall, Function* indirectFunction) {
  auto calledFunction = cast<CallInst>(patchedDirectCall)->getCalledFunction();
  auto entryBlock = &calledFunction->getEntryBlock();

  // Get the fixp call instruction to use it as an argument for the restored
  // library function
  auto fixpCallInstr = entryBlock->getTerminator()->getPrevNode();
  assert(isa<CallInst>(fixpCallInstr) && "expected a CallInst to the outlined function");
  auto fixpCall = cast<CallInst>(fixpCallInstr);

  // Use bitcast to keep compatibility with the OpenMP runtime reference
  auto microTaskType = indirectFunction->getArg(2)->getType();
  auto bitcastedMicroTask = ConstantExpr::getBitCast(fixpCall->getCalledFunction(), microTaskType);

  // Add the indirect arguments
  std::vector<Value*> indirectCallArgs =
    std::vector<Value*>(patchedDirectCall->arg_begin(), patchedDirectCall->arg_end());
  indirectCallArgs.insert(indirectCallArgs.begin() + 2, bitcastedMicroTask);

  // Insert the indirect call after the patched direct call
  auto indirectCall = CallInst::Create(indirectFunction, indirectCallArgs);
  indirectCall->insertAfter(patchedDirectCall);
  copyValueInfo(indirectCall, patchedDirectCall);

  // Remove the patched direct call
  TaffoInfo::getInstance().eraseValue(patchedDirectCall);
}
