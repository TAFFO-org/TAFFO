#include "NameVariables.h"

#include "Metadata.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "TaffoUtils/TypeUtils.h"

using namespace llvm;

#define DEBUG_TYPE "name-variables"


bool NameVariables::runOnModule(Module &M) {
  bool ChangedVarNames = false;

  auto &CTX = M.getContext();
  IRBuilder<> Builder(CTX);

  std::list<Function*> funcs{};

  for (auto &F : M) {
    funcs.push_back(&F);
  }

  while (!funcs.empty()) {
    auto *Fptr = funcs.front();
    auto &F = *Fptr;
    funcs.pop_front();
    if (!F.hasName() || F.isDeclaration() || !isFPFunction(&F)
//        || !(
//               F.getName() == "_Z19BlkSchlsEqEuroNoDivfffffifPfS_"
//            || F.getName() == "_Z4CNDFf"
////            || F.getName() == "_ZSt3expf"
////            || F.getName() == "_ZSt4sqrtf"
//            || F.getName() == "_ZSt3logf"
//             )
        ) {
      continue;
    }

    errs() << F.getName() << "\n";

    while (!F.users().empty()) {
      auto *user = *F.users().begin();
      if (isa<CallInst>(user) || isa<InvokeInst>(user)) {
        auto *call = dyn_cast<CallBase>(user);
        errs() << *call << "\n";

        Function *newF = createFunctionCopy(call);
        call->setCalledFunction(newF);

        // Attach metadata
        MDNode *newFRef = MDNode::get(call->getContext(), ValueAsMetadata::get(newF));
        MDNode *oldFRef = MDNode::get(call->getContext(), ValueAsMetadata::get(&F));

        call->setMetadata(ORIGINAL_FUN_METADATA, oldFRef);
        if (MDNode *cloned = F.getMetadata(CLONED_FUN_METADATA)) {
          cloned = cloned->concatenate(cloned, newFRef);
          F.setMetadata(CLONED_FUN_METADATA, cloned);
        } else {
          F.setMetadata(CLONED_FUN_METADATA, newFRef);
        }
        newF->setMetadata(CLONED_FUN_METADATA, NULL);
        newF->setMetadata(SOURCE_FUN_METADATA, oldFRef);
      }
    }
  }

  long counter = 0;
  auto moduleName = "a";

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
          ChangedVarNames = true;
        }
        current = next;
      }
    }
  }

  return ChangedVarNames;
}

bool NameVariables::isFPFunction(llvm::Function *F)
{
  bool result = taffo::isFloatType(F->getReturnType());
  for (auto &arg: F->args()) {
    result = result || taffo::isFloatType(arg.getType());
  }
  return result;
}

llvm::Function *NameVariables::createFunctionCopy(llvm::CallBase *call)
{
  Function *oldF = call->getCalledFunction();
  Function *newF = Function::Create(
      oldF->getFunctionType(), oldF->getLinkage(),
      oldF->getName(), oldF->getParent());

  ValueToValueMapTy mapArgs; // Create Val2Val mapping and clone function
  Function::arg_iterator newArgumentI = newF->arg_begin();
  Function::arg_iterator oldArgumentI = oldF->arg_begin();
  for (; oldArgumentI != oldF->arg_end(); oldArgumentI++, newArgumentI++) {
    newArgumentI->setName(oldArgumentI->getName());
    mapArgs.insert(std::make_pair(oldArgumentI, newArgumentI));
  }
  SmallVector<ReturnInst *, 100> returns;
  CloneFunctionInto(newF, oldF, mapArgs, true, returns);
  newF->setLinkage(GlobalVariable::LinkageTypes::InternalLinkage);
  return newF;
}

PreservedAnalyses NameVariables::run(llvm::Module &M,
                                      llvm::ModuleAnalysisManager &) {
  bool Changed =  runOnModule(M);

  return (Changed ? llvm::PreservedAnalyses::none()
                  : llvm::PreservedAnalyses::all());
}
