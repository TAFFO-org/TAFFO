#include "MemToRegPass.hpp"
#include "PromoteMemToReg.hpp"
#include "TaffoInfo/TaffoInfo.hpp"

#include <llvm/ADT/Statistic.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Casting.h>

using namespace llvm;
using namespace taffo;

#define DEBUG_TYPE "taffo-mem2reg"

STATISTIC(numPromoted, "Number of alloca's promoted");

bool MemToRegPass::promoteMemoryToRegister(Function& f,
                                           DominatorTree& dominatorTree,
                                           AssumptionCache& assumptionCache) {
  SmallVector<AllocaInst*> allocas;
  BasicBlock& bb = f.getEntryBlock(); // Get the entry node for the function
  bool changed = false;

  while (true) {
    allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in the entry node
    for (Instruction& inst : bb)
      if (AllocaInst* allocaInst = dyn_cast<AllocaInst>(&inst))
        if (isAllocaPromotable(allocaInst))
          allocas.push_back(allocaInst);

    if (allocas.empty())
      break;

    promoteMemToReg(allocas, dominatorTree, &assumptionCache);
    numPromoted += allocas.size();
    changed = true;
  }
  return changed;
}

PreservedAnalyses MemToRegPass::run(Function& f, FunctionAnalysisManager& analysisManager) {
  static bool initializedTaffoInfo = false;

  Module& m = *f.getParent();
  if (!initializedTaffoInfo) {
    TaffoInfo::getInstance().initializeFromFile("taffo_info_init.json", m);
    initializedTaffoInfo = true;
  }

  auto& dominatorTree = analysisManager.getResult<DominatorTreeAnalysis>(f);
  auto& assumptionCache = analysisManager.getResult<AssumptionAnalysis>(f);
  if (!promoteMemoryToRegister(f, dominatorTree, assumptionCache))
    return PreservedAnalyses::all();

  TaffoInfo::getInstance().dumpToFile("taffo_info_memToReg.json", m);

  PreservedAnalyses preservedAnalyses;
  preservedAnalyses.preserveSet<CFGAnalyses>();
  return preservedAnalyses;
}
