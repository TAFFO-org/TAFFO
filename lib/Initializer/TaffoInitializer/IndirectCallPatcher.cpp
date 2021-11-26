#include <unordered_set>
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "IndirectCallPatcher.h"
#include "Metadata.h"
#include "TaffoInitializerPass.h"
#include "TypeUtils.h"

using namespace taffo;
using namespace llvm;

/// Check recursively whether an unsupported function is called.
bool containsUnsupportedFunctions(
    const llvm::Function *function,
    std::unordered_set<Function *> traversedFunctions)
{
  static const std::vector<std::string> prefixBlocklist{"__kmpc_omp_task",
                                                        "__kmpc_reduce"};

  for (auto instructionIt = inst_begin(function);
       instructionIt != inst_end(function); instructionIt++) {
    if (auto curCallInstruction = dyn_cast<CallInst>(&(*instructionIt))) {
      llvm::Function *curCallFunction = curCallInstruction->getCalledFunction();
      auto functionName = curCallFunction->getName();

      if (any_of(prefixBlocklist, [&](const std::string &prefix) {
            return functionName.startswith(prefix);
          })) {
        return true;
      } else if (traversedFunctions.find(curCallFunction) ==
                 traversedFunctions.end()) {
        traversedFunctions.insert(curCallFunction);
        if (containsUnsupportedFunctions(curCallFunction, traversedFunctions)) {
          return true;
        }
      }
    }
  }
  return false;
}

/// Handle the __kmpc_fork_call replacing the indirect call with a direct call.
/// In case an unsupported function is called, keep the indirect function and
/// attach the OMP disabled metadata to the the shared variables.
void handleKmpcFork(const Module &m, std::vector<Instruction *> &toDelete,
                    CallInst *curCallInstruction, const CallSite *curCall,
                    Function *indirectFunction)
{

  auto microTaskOperand =
      dyn_cast<ConstantExpr>(curCall->arg_begin() + 2)->getOperand(0);
  auto microTaskFunction = dyn_cast_or_null<Function>(microTaskOperand);

  assert(
          microTaskFunction != nullptr &&
          "The microtask function must be present in the __kmpc_fork_call as a "
          "third argument");

  if (containsUnsupportedFunctions(microTaskFunction, {})) {
    LLVM_DEBUG(dbgs() << "Blocking conversion for shared variables in "
                         "unsupported parallel region"
                      << *curCallInstruction << "\n");

    auto *constantPointerNull =
        ConstantPointerNull::get(Type::getInt32PtrTy(m.getContext()));
    MDNode *metadataNode =
        MDNode::get(m.getContext(), ValueAsMetadata::get(constantPointerNull));

    for (auto *sharedArgument = curCall->arg_begin() + 2;
         sharedArgument < curCall->arg_end(); sharedArgument++)
      if (auto *sharedVarInstr = dyn_cast<Instruction>(*sharedArgument))
        sharedVarInstr->setMetadata(OMP_DISABLED_METADATA, metadataNode);
  };

  std::vector<Type *> paramsFunc;

  auto functionType = indirectFunction->getFunctionType();

  auto params = functionType->params();
  // Copy the first two params directly from the functionType since they fixed
  // internal OpenMP parameters
  copy_n(params.begin(), 2, back_inserter(paramsFunc));
  // Skip the third argument (outlined function) and copy the dynamic arguments'
  // types from the call
  for (unsigned i = 3; i < curCall->getNumArgOperands(); i++)
    paramsFunc.push_back(curCall->getArgOperand(i)->getType());

  // Create the new function with the parsed types and signature
  auto trampolineFunctionType =
      FunctionType::get(functionType->getReturnType(), paramsFunc, false);
  auto trampolineFunctionName = indirectFunction->getName() + "_trampoline";
  Function *trampolineFunction =
      Function::Create(trampolineFunctionType, indirectFunction->getLinkage(),
                       trampolineFunctionName, indirectFunction->getParent());

  // Shift back the argument name since the third argument is skipped
  for (unsigned i = 3; i < curCall->getNumArgOperands(); i++) {
    trampolineFunction->getArg(i - 1)->setName(
        curCall->getArgOperand(i)->getName());
  }

  BasicBlock *block =
      BasicBlock::Create(m.getContext(), "main", trampolineFunction);

  // Create the arguments of the trampoline function from the original call,
  // skipping the third
  std::vector<Value *> trampolineArgs;
  copy_n(curCall->arg_begin(), 2, back_inserter(trampolineArgs));
  copy(curCall->arg_begin() + 3, curCall->arg_end(),
       back_inserter(trampolineArgs));

  // Keep ref to the indirect function, preventing globaldce pass to destroy it
  auto magicBitCast =
      new BitCastInst(indirectFunction, indirectFunction->getType(), "", block);
  ReturnInst::Create(m.getContext(), nullptr, block);

  // Create the arguments of the direct function, extracted from the indirect
  std::vector<Value *> outlinedArgumentsInsideTrampoline;
  // Create null pointer to patch the internal OpenMP argument
  Value *nullPointer = ConstantPointerNull::get(
      PointerType::get(Type::getInt32Ty(trampolineFunction->getContext()), 0));
  outlinedArgumentsInsideTrampoline.push_back(nullPointer);
  outlinedArgumentsInsideTrampoline.push_back(nullPointer);
  for (auto argIt = trampolineFunction->arg_begin() + 2;
       argIt < trampolineFunction->arg_end(); argIt++) {
    outlinedArgumentsInsideTrampoline.push_back(argIt);
  }

  // Create the call to the direct function inside the trampoline
  CallInst *outlinedCall =
      CallInst::Create(microTaskFunction, outlinedArgumentsInsideTrampoline);
  outlinedCall->insertAfter(magicBitCast);

  // Create the call to the trampoline function after the indirect function
  CallInst *trampolineCallInstruction =
      CallInst::Create(trampolineFunction, trampolineArgs);
  trampolineCallInstruction->setCallingConv(
      curCallInstruction->getCallingConv());
  trampolineCallInstruction->insertBefore(curCallInstruction);
  trampolineCallInstruction->setDebugLoc(curCallInstruction->getDebugLoc());

  MDNode *indirectFunctionRef = MDNode::get(
      curCallInstruction->getContext(), ValueAsMetadata::get(indirectFunction));
  trampolineCallInstruction->setMetadata(INDIRECT_METADATA,
                                         indirectFunctionRef);

  // Save the old instruction to delete it later
  toDelete.push_back(curCallInstruction);
  LLVM_DEBUG(dbgs() << "Newly created instruction: "
                    << *trampolineCallInstruction << "\n");
}

/// Check if the given call is indirect and handle it with the dedicated handler.
void handleIndirectCall(const Module &m, std::vector<Instruction *> &toDelete,
                        CallInst *curCallInstruction, const CallBase *curCall,
                        Function *indirectFunction)
{
  using handler_function = void (*)(const llvm::Module &m,
                                    std::vector<llvm::Instruction *> &toDelete,
                                    llvm::CallInst *curCallInstruction,
                                    const CallSite *curCall,
                                    llvm::Function *indirectFunction);
  const static std::map<const std::string, handler_function> indirectCallFunctions = {
          {"__kmpc_fork_call", &handleKmpcFork}
  };

  auto indirectCallHandler =
      indirectCallFunctions.find((std::string)indirectFunction->getName());

  if (indirectCallHandler != indirectCallFunctions.end())
    indirectCallHandler->second(m, toDelete, curCallInstruction, curCall,
                                indirectFunction);
}

/// Check the indirect calls in the given module, and handle them with handleIndirectCall().
void taffo::manageIndirectCalls(llvm::Module &m)
{
  LLVM_DEBUG(dbgs() << "Checking Indirect Calls"
                    << "\n");

  std::vector<Instruction *> toDelete;

  for (llvm::Function &curFunction : m) {
    for (auto instructionIt = inst_begin(curFunction);
         instructionIt != inst_end(curFunction); instructionIt++) {
      if (auto curCallInstruction = dyn_cast<CallInst>(&(*instructionIt))) {

        llvm::Function *curCallFunction = curCallInstruction->getCalledFunction();

        if (curCallFunction) {
          handleIndirectCall(m, toDelete, curCallInstruction, dyn_cast<CallBase>(curCallInstruction),
                             curCallFunction);
        }
      }
    }
  }

  // Delete the saved instructions in a separate loop to avoid conflicts in the
  // iterator
  for (auto inst : toDelete) {
    inst->eraseFromParent();
  }
}
