#include "StripAnnotations.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "strip-annotations"

bool StripAnnotations::runOnModule(llvm::Module &M)
{
  bool Changed = false;
  auto &CTX = M.getContext();
  IRBuilder<> Builder(CTX);

  for (auto &F : M) {
    if (!F.hasName() || F.isDeclaration())
      continue;

    for (auto &BB: F.getBasicBlockList()) {
      auto &InstList = BB.getInstList();
      auto current = InstList.getNextNode(InstList.front());
      while (current != nullptr) {
        auto &Inst = *current;
        auto next = InstList.getNextNode(*current);
        if (!Inst.isDebugOrPseudoInst()) {
          if (auto *call = dyn_cast<CallInst>(current)) {
            if (call->getCalledFunction() && call->getCalledFunction()->getName() == "llvm.var.annotation") {
              errs() << "removing: " << *call << "\n";
              call->eraseFromParent();
              Changed = true;
            }
          }
        }
        current = next;
      }
    }
  }

  GlobalVariable *globAnnos = M.getGlobalVariable("llvm.global.annotations");
  if (globAnnos != nullptr) {
    errs() << "removing: " << *globAnnos << "\n";
    globAnnos->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

PreservedAnalyses StripAnnotations::run(llvm::Module &M,
                                      llvm::ModuleAnalysisManager &) {
  bool Changed =  runOnModule(M);

  return (Changed ? llvm::PreservedAnalyses::none()
                  : llvm::PreservedAnalyses::all());
}
