#include "FloatSizeAnalysis.h"

#include "llvm/IR/IRBuilder.h"
#include "Metadata.h"

using namespace llvm;

#define DEBUG_TYPE "float-size-analysis"

bool FloatSizeAnalysis::runOnModule(llvm::Module &M)
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
          auto* binOp = dyn_cast<BinaryOperator>(&Inst);
          if (binOp) {
            auto opCode = binOp->getOpcode();
            switch (opCode) {
            case llvm::Instruction::FMul:
            case llvm::Instruction::FAdd:
            case llvm::Instruction::FDiv:
            case llvm::Instruction::FSub:
              printOpRanges(binOp);
              break ;
            default:
              break ;
            }
          }
        }
        current = next;
      }
    }
  }

  return Changed;
}

void FloatSizeAnalysis::printOpRanges(BinaryOperator *binOp) {
  auto op0 = binOp->getOperand(0);
  auto op1 = binOp->getOperand(1);
  mdutils::MDInfo *mdiOp0 = mdutils::MetadataManager::getMetadataManager().retrieveMDInfo(op0);
  mdutils::MDInfo *mdiOp1 = mdutils::MetadataManager::getMetadataManager().retrieveMDInfo(op1);
  if (mdiOp0 && mdiOp1) {
    auto* iiop0 = dyn_cast<mdutils::InputInfo>(mdiOp0);
    auto* iiop1 = dyn_cast<mdutils::InputInfo>(mdiOp1);
    if (iiop0 && iiop1 && iiop0->IRange && iiop1->IRange) {
      errs() << "-----\n";
      errs() << *binOp << "\n";
      errs() << "op0: (" << iiop0->IRange->Min << ", " << iiop0->IRange->Max << ")\n";
      errs() << "op1: (" << iiop1->IRange->Min << ", " << iiop1->IRange->Max << ")\n";
    }
  }
}

PreservedAnalyses FloatSizeAnalysis::run(llvm::Module &M,
                                        llvm::ModuleAnalysisManager &) {
  bool Changed =  runOnModule(M);

  return (Changed ? llvm::PreservedAnalyses::none()
                  : llvm::PreservedAnalyses::all());
}
