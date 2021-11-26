#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "LLVMFloatToFixedPass.h"
#include "TypeUtils.h"

using namespace llvm;
using namespace taffo;
using namespace flttofix;

/// Retrieve the indirect calls converted into trampolines and re-use the
/// original indirect functions.
void FloatToFixed::convertIndirectCalls(llvm::Module &m)
{
  using handler_function = void (FloatToFixed::*)(
          llvm::CallInst *patchedDirectCall, llvm::Function *indirectFunction);
  const std::map<const std::string, handler_function> indirectCallFunctions = {
          {"__kmpc_fork_call", &FloatToFixed::handleKmpcFork}};

  std::vector<llvm::CallInst *> trampolineCalls;

  // Retrieve the trampoline calls using the INDIRECT_METADATA
  for (llvm::Function &curFunction : m) {
    for (auto instructionIt = inst_begin(curFunction);
         instructionIt != inst_end(curFunction); instructionIt++) {
      if (auto curCallInstruction =
              llvm::dyn_cast<llvm::CallInst>(&(*instructionIt))) {
        if (curCallInstruction->getMetadata(INDIRECT_METADATA)) {
          trampolineCalls.push_back(curCallInstruction);
        }
      }
    }
  }

  // Convert the trampoline calls
  for (auto trampolineCall : trampolineCalls) {
    auto *metadata = dyn_cast<ValueAsMetadata>(
        *trampolineCall->getMetadata(INDIRECT_METADATA)->op_begin());
    auto *indirectFunction = dyn_cast_or_null<Function>(metadata->getValue());

    if (indirectFunction == nullptr) {
      LLVM_DEBUG(dbgs() << "Blocking the following conversion for failed "
                           "dyn_cast on the indirect function: "
                        << *trampolineCall << "\n");
      continue;
    }

    auto indirectCallHandler =
        indirectCallFunctions.find((std::string)indirectFunction->getName());

    if (indirectCallHandler != indirectCallFunctions.end()) {
      handler_function indirectFunctionHandler = indirectCallHandler->second;
      (this->*indirectFunctionHandler)(trampolineCall, indirectFunction);
    }
  }
}

/// Convert a trampoline call to an outlined function back into the original
/// library function
void FloatToFixed::handleKmpcFork(CallInst *patchedDirectCall,
                                  Function *indirectFunction)
{
  auto calledFunction = cast<CallInst>(patchedDirectCall)->getCalledFunction();
  auto entryBlock = &calledFunction->getEntryBlock();

  // Get the fixp call instruction to use it as an argument for the restored
  // library function
  auto fixpCallInstr = entryBlock->getTerminator()->getPrevNode();
  assert(llvm::isa<llvm::CallInst>(fixpCallInstr) &&
         "expected a CallInst to the outlined function");
  auto fixpCall = cast<CallInst>(fixpCallInstr);

  // Use bitcast to keep compatibility with the OpenMP runtime reference
  auto microTaskType = indirectFunction->getArg(2)->getType();
  auto bitcastedMicroTask =
      ConstantExpr::getBitCast(fixpCall->getCalledFunction(), microTaskType);

  // Add the indirect arguments
  std::vector<Value *> indirectCallArgs = std::vector<Value *>(
      patchedDirectCall->arg_begin(), patchedDirectCall->arg_end());
  indirectCallArgs.insert(indirectCallArgs.begin() + 2, bitcastedMicroTask);

  // Insert the indirect call after the patched direct call
  auto indirectCall = CallInst::Create(indirectFunction, indirectCallArgs);
  indirectCall->insertAfter(patchedDirectCall);
  cpMetaData(indirectCall, patchedDirectCall);

  // Remove the patched direct call
  patchedDirectCall->eraseFromParent();
}
