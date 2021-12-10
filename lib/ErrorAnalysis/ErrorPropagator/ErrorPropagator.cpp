//===-- ErrorPropagator.cpp - Error Propagator ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This LLVM opt pass propagates errors in fixed point computations.
///
//===----------------------------------------------------------------------===//

#include "ErrorPropagator.h"

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/Debug.h"

#include "FunctionErrorPropagator.h"
#include "Metadata.h"

namespace ErrorProp
{

using namespace llvm;
using namespace mdutils;

#define DEBUG_TYPE "errorprop"

bool ErrorPropagator::runOnModule(Module &M)
{
  checkCommandLine();

  MetadataManager &MDManager = MetadataManager::getMetadataManager();

  RangeErrorMap GlobalRMap(MDManager, !Relative, ExactConst);

  // Get Ranges and initial Errors for global variables.
  retrieveGlobalVariablesRangeError(M, GlobalRMap);

  // Copy list of original functions, so we don't mess up with copies.
  SmallVector<Function *, 4U> Functions;
  Functions.reserve(M.size());
  for (Function &F : M) {
    Functions.push_back(&F);
  }

  FunctionCopyManager FCMap(*this, MaxRecursionCount, DefaultUnrollCount,
                            MaxUnroll);

  bool NoFunctions = true;
  // Iterate over all functions in this Module,
  // and propagate errors for pending input intervals for all of them.
  for (Function *F : Functions) {
    if (StartOnly && !MetadataManager::isStartingPoint(*F))
      continue;

    NoFunctions = false;
    FunctionErrorPropagator FEP(*this, *F, FCMap, MDManager, SloppyAA);
    FEP.computeErrorsWithCopy(GlobalRMap, nullptr, true);
  }

  if (NoFunctions)
    dbgs() << "[taffo-err] WARNING: no starting-point functions found. Try running taffo-err without -startonly.\n";

  dbgs() << "\n*** Target Errors: ***\n";
  GlobalRMap.printTargetErrors(dbgs());

  return false;
}

void ErrorPropagator::retrieveGlobalVariablesRangeError(Module &M,
                                                        RangeErrorMap &RMap)
{
  for (GlobalVariable &GV : M.globals()) {
    RMap.retrieveRangeError(GV);
  }
}

void ErrorPropagator::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.addRequiredTransitive<DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<LoopInfoWrapperPass>();
  AU.addRequiredTransitive<AssumptionCacheTracker>();
  AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
  AU.addRequiredTransitive<OptimizationRemarkEmitterWrapperPass>();
  AU.addRequiredTransitive<MemorySSAWrapperPass>();
  AU.addRequiredTransitive<TargetTransformInfoWrapperPass>();
  AU.setPreservesAll();
}

void ErrorPropagator::checkCommandLine()
{
  if (CmpErrorThreshold > 100U)
    CmpErrorThreshold = 100U;

  if (NoLoopUnroll)
    MaxUnroll = 0U;
}

} // end of namespace ErrorProp

char ErrorProp::ErrorPropagator::ID = 0;

static llvm::RegisterPass<ErrorProp::ErrorPropagator>
    X("errorprop", "Fixed-Point Arithmetic Error Propagator",
      false /* Only looks at CFG */,
      false /* Analysis Pass */);
