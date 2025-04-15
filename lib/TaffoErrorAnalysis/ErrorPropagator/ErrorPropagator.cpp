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

#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/MemorySSA.h>
#include <llvm/Analysis/OptimizationRemarkEmitter.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Support/Debug.h>

#include "FunctionErrorPropagator.h"
#include "Metadata.h"

#define DEBUG_TYPE "errorprop"

namespace ErrorProp
{

using namespace llvm;
using namespace mdutils;

cl::opt<unsigned> DefaultUnrollCount("dunroll",
                                     cl::desc("Default loop unroll count"),
                                     cl::value_desc("count"),
                                     cl::init(1U));
cl::opt<unsigned> MaxUnroll("maxunroll",
                            cl::desc("Max loop unroll count. "
                                     "Setting this to 0 disables loop unrolling. "
                                     "(Default: 256)"),
                            cl::value_desc("count"),
                            cl::init(256U));
cl::opt<bool> NoLoopUnroll("nounroll",
                           cl::desc("Never unroll loops (legacy, use -max-unroll=0)"),
                           cl::init(false));
cl::opt<unsigned> CmpErrorThreshold("cmpthresh",
                                    cl::desc("CMP errors are signaled"
                                             "only if error is above perc %"),
                                    cl::value_desc("perc"),
                                    cl::init(0U));
cl::opt<unsigned> MaxRecursionCount("recur",
                                    cl::desc("Default number of recursive calls"
                                             "to the same function."),
                                    cl::value_desc("count"),
                                    cl::init(1U));
cl::opt<bool> StartOnly("startonly",
                        cl::desc("Propagate only functions with start metadata."),
                        cl::init(false));
cl::opt<bool> Relative("relerror",
                       cl::desc("Output relative errors instead of absolute errors (experimental)."),
                       cl::init(false));
cl::opt<bool> ExactConst("exactconst",
                         cl::desc("Treat all constants as exact."),
                         cl::init(false));
cl::opt<bool> SloppyAA("sloppyaa",
                       cl::desc("Enable sloppy Alias Analysis, for when LLVM AA fails."),
                       cl::init(false));

PreservedAnalyses ErrorPropagator::run(Module &M, ModuleAnalysisManager &AM)
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

  FunctionCopyManager FCMap(AM, MaxRecursionCount, DefaultUnrollCount,
                            MaxUnroll);

  FunctionAnalysisManager &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  bool NoFunctions = true;
  // Iterate over all functions in this Module,
  // and propagate errors for pending input intervals for all of them.
  for (Function *F : Functions) {
    if (StartOnly && !MetadataManager::isStartingPoint(*F))
      continue;

    NoFunctions = false;
    FunctionErrorPropagator FEP(FAM, *F, FCMap, MDManager, SloppyAA);
    FEP.computeErrorsWithCopy(GlobalRMap, nullptr, true);
  }

  if (NoFunctions)
    dbgs() << "[taffo-err] WARNING: no starting-point functions found. Try running taffo-err without -startonly.\n";

  dbgs() << "\n*** Target Errors: ***\n";
  GlobalRMap.printTargetErrors(dbgs());

  return PreservedAnalyses::none();
}

void ErrorPropagator::retrieveGlobalVariablesRangeError(Module &M,
                                                        RangeErrorMap &RMap)
{
  for (GlobalVariable &GV : M.globals()) {
    RMap.retrieveRangeError(GV);
  }
}

void ErrorPropagator::checkCommandLine()
{
  if (CmpErrorThreshold > 100U)
    CmpErrorThreshold = 100U;

  if (NoLoopUnroll)
    MaxUnroll = 0U;
}

} // end of namespace ErrorProp
