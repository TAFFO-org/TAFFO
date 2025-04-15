//===- Mem2Reg.cpp - The -mem2reg pass, a wrapper around the Utils lib ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//

#include "MemToRegPass.hpp"

#include "TaffoInfo/TaffoInfo.hpp"
#include "PromoteMemToReg.hpp"

#include <llvm/ADT/Statistic.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Casting.h>
#include <vector>

using namespace llvm;
using namespace taffo;

#define DEBUG_TYPE "taffo-mem2reg"

STATISTIC(NumPromoted, "Number of alloca's promoted");

static bool promoteMemoryToRegister(Function &F, DominatorTree &DT, AssumptionCache &AC) {
  std::vector<AllocaInst *> Allocas;
  BasicBlock &BB = F.getEntryBlock(); // Get the entry node for the function
  bool Changed = false;

  while (true) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) // Is it an alloca?
        if (isAllocaPromotable(AI))
          Allocas.push_back(AI);

    if (Allocas.empty())
      break;

    PromoteMemToReg(Allocas, DT, &AC);
    NumPromoted += Allocas.size();
    Changed = true;
  }
  return Changed;
}

PreservedAnalyses MemToRegPass::run(Function &F, FunctionAnalysisManager &AM) {
  static bool initializedTaffoInfo = false;

  Module &M = *F.getParent();
  if (!initializedTaffoInfo) {
    TaffoInfo::getInstance().initializeFromFile("taffo_info_init.json", M);
    initializedTaffoInfo = true;
  }

  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  if (!promoteMemoryToRegister(F, DT, AC))
    return PreservedAnalyses::all();

  TaffoInfo::getInstance().dumpToFile("taffo_info_memToReg.json", M);

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
