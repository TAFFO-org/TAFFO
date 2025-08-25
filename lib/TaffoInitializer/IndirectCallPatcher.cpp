#include "Debug/Logger.hpp"
#include "InitializerPass.hpp"
#include "TaffoInfo/TaffoInfo.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <complex>
#include <map>
#include <unordered_set>

using namespace taffo;
using namespace tda;
using namespace llvm;

#define DEBUG_TYPE "taffo-init"

bool containsUnsupportedFunction(const Function* function, std::unordered_set<Function*> visitedFunctions) {
  static const std::vector<std::string> prefixBlocklist {"__kmpc_omp_task", "__kmpc_reduce"};
  for (auto& inst : instructions(function)) {
    if (auto call = dyn_cast<CallInst>(&inst)) {
      Function* calledFunction = call->getCalledFunction();
      auto funName = calledFunction->getName();
      if (any_of(prefixBlocklist, [&](const std::string& prefix) { return funName.starts_with(prefix); }))
        return true;
      if (!visitedFunctions.contains(calledFunction)) {
        visitedFunctions.insert(calledFunction);
        if (containsUnsupportedFunction(calledFunction, visitedFunctions))
          return true;
      }
    }
  }
  return false;
}

void InitializerPass::handleKmpcFork(const Module& m, CallBase* call, Function* indirectFunction) {
  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger << "[" << __FUNCTION__ << "] ";
    logger.logValueln(call);
    indenter.increaseIndent());

  auto microtaskFunction = dyn_cast_or_null<Function>(call->getArgOperand(2));
  assert(microtaskFunction != nullptr
         && "The microtask function must be present in the __kmpc_fork_call as third argument");

  if (containsUnsupportedFunction(microtaskFunction, {})) {
    LLVM_DEBUG(log() << "unsupported parallel region: blocking conversion of shared variables\n");
    for (auto* sharedArg = call->arg_begin() + 2; sharedArg < call->arg_end(); sharedArg++)
      if (auto* sharedVarInstr = dyn_cast<Instruction>(*sharedArg))
        taffoInfo.createValueInfo(*sharedVarInstr)->disableConversion();
  }

  TransparentType* retType = taffoInfo.getOrCreateTransparentType(*indirectFunction);
  SmallVector<TransparentType*, 8> argTypes;
  // Copy the first two args directly from the function since they are fixed internal OpenMP parameters
  unsigned i = 0;
  for (; i < 2; i++) {
    Argument* funArg = indirectFunction->getArg(i);
    argTypes.push_back(taffoInfo.getOrCreateTransparentType(*funArg));
  }
  // Skip the third argument (outlined function) and copy the dynamic arguments' types from the call
  i++;
  for (; i < call->arg_size(); i++) {
    Value* callArg = call->getArgOperand(i);
    argTypes.push_back(taffoInfo.getOrCreateTransparentType(*callArg));
  }
  auto argLLVMTypes = llvm::to_vector<8>(map_range(argTypes, [](const auto& t) { return t->toLLVMType(); }));

  // Create the new function with the parsed types and signature
  FunctionType* trampolineFunType = FunctionType::get(retType->toLLVMType(), argLLVMTypes, false);
  Function* trampolineFunction = Function::Create(trampolineFunType,
                                                  indirectFunction->getLinkage(),
                                                  indirectFunction->getName() + "_trampoline",
                                                  indirectFunction->getParent());

  // Copy transparent types to the new function
  taffoInfo.setTransparentType(*trampolineFunction, retType->clone());
  for (auto&& [arg, type] : zip(trampolineFunction->args(), argTypes))
    taffoInfo.setTransparentType(arg, type->clone());

  // Shift back the argument name since the third argument is skipped
  for (unsigned i = 3; i < call->arg_size(); i++)
    trampolineFunction->getArg(i - 1)->setName(call->getArgOperand(i)->getName());

  BasicBlock* bb = BasicBlock::Create(m.getContext(), "main", trampolineFunction);

  // Create the arguments of the trampoline function from the original call, skipping the third
  std::vector<Value*> trampolineCallArgs;
  copy_n(call->arg_begin(), 2, back_inserter(trampolineCallArgs));
  copy(call->arg_begin() + 3, call->arg_end(), back_inserter(trampolineCallArgs));

  // Keep ref to the indirect function, preventing globaldce pass to destroy it
  auto magicBitCast = new BitCastInst(indirectFunction, indirectFunction->getType(), "", bb);
  ReturnInst::Create(m.getContext(), nullptr, bb);

  // Create the arguments of the direct function, extracted from the indirect
  std::vector<Value*> outlinedCallArgsInsideTrampoline;
  // Create null pointer to patch the internal OpenMP argument
  Value* nullPointer = ConstantPointerNull::get(PointerType::get(trampolineFunction->getContext(), 0));

  outlinedCallArgsInsideTrampoline.push_back(nullPointer);
  outlinedCallArgsInsideTrampoline.push_back(nullPointer);
  for (auto argIter = trampolineFunction->arg_begin() + 2; argIter < trampolineFunction->arg_end(); argIter++)
    outlinedCallArgsInsideTrampoline.push_back(argIter);

  // Create the call to the direct function inside the trampoline
  CallInst* outlinedCall = CallInst::Create(microtaskFunction, outlinedCallArgsInsideTrampoline);
  outlinedCall->insertAfter(magicBitCast);

  // Create the call to the trampoline function after the indirect function
  CallInst* trampolineCall = CallInst::Create(trampolineFunction, trampolineCallArgs);
  trampolineCall->setCallingConv(call->getCallingConv());
  trampolineCall->insertBefore(call);
  trampolineCall->setDebugLoc(call->getDebugLoc());

  taffoInfo.setIndirectFunction(*trampolineCall, *indirectFunction);
  taffoInfo.eraseValue(call);
  LLVM_DEBUG(log() << "trampoline call: " << *trampolineCall << "\n");
}

void InitializerPass::handleCallIfIndirect(const Module& m, CallBase* call, Function* calledFunction) {
  using IndirectCallHandler = void (InitializerPass::*)(const Module& m, CallBase* call, Function* calledFunction);
  const std::map<const std::string, IndirectCallHandler> indirectCallHandlers = {
    {"__kmpc_fork_call", &InitializerPass::handleKmpcFork}
  };

  auto iter = indirectCallHandlers.find(static_cast<std::string>(calledFunction->getName()));
  if (iter != indirectCallHandlers.end())
    (this->*iter->second)(m, call, calledFunction);
}

void InitializerPass::manageIndirectCalls(Module& m) {
  LLVM_DEBUG(log().logln("[Checking indirect calls]", Logger::Blue));
  for (Function& f : m)
    for (Instruction& inst : make_early_inc_range(instructions(f)))
      if (auto* call = dyn_cast<CallBase>(&inst))
        if (Function* calledFunction = call->getCalledFunction())
          handleCallIfIndirect(m, call, calledFunction);
}
