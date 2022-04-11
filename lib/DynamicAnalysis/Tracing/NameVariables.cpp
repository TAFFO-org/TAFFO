#include "NameVariables.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;

#define DEBUG_TYPE "name-variables"


bool NameVariables::runOnModule(Module &M) {
  bool ChangedVarNames = false;

  auto &CTX = M.getContext();
  IRBuilder<> Builder(CTX);
  long counter = 0;
  auto moduleName = M.getModuleIdentifier();

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

PreservedAnalyses NameVariables::run(llvm::Module &M,
                                      llvm::ModuleAnalysisManager &) {
  bool Changed =  runOnModule(M);

  return (Changed ? llvm::PreservedAnalyses::none()
                  : llvm::PreservedAnalyses::all());
}
