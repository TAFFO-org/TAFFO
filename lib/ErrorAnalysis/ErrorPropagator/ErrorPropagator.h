//===-- ErrorPropagator.h - Error Propagator --------------------*- C++ -*-===//
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

#ifndef ERRORPROPAGATOR_H
#define ERRORPROPAGATOR_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

#include "FunctionCopyMap.h"
#include "RangeErrorMap.h"

#define DEBUG_TYPE "errorprop"

namespace ErrorProp
{

llvm::cl::opt<unsigned> DefaultUnrollCount("dunroll",
                                           llvm::cl::desc("Default loop unroll count"),
                                           llvm::cl::value_desc("count"),
                                           llvm::cl::init(1U));
llvm::cl::opt<unsigned> MaxUnroll("maxunroll",
                                  llvm::cl::desc("Max loop unroll count. "
                                                 "Setting this to 0 disables loop unrolling. "
                                                 "(Default: 256)"),
                                  llvm::cl::value_desc("count"),
                                  llvm::cl::init(256U));
llvm::cl::opt<bool> NoLoopUnroll("nounroll",
                                 llvm::cl::desc("Never unroll loops (legacy, use -max-unroll=0)"),
                                 llvm::cl::init(false));
llvm::cl::opt<unsigned> CmpErrorThreshold("cmpthresh",
                                          llvm::cl::desc("CMP errors are signaled"
                                                         "only if error is above perc %"),
                                          llvm::cl::value_desc("perc"),
                                          llvm::cl::init(0U));
llvm::cl::opt<unsigned> MaxRecursionCount("recur",
                                          llvm::cl::desc("Default number of recursive calls"
                                                         "to the same function."),
                                          llvm::cl::value_desc("count"),
                                          llvm::cl::init(1U));
llvm::cl::opt<bool> StartOnly("startonly",
                              llvm::cl::desc("Propagate only functions with start metadata."),
                              llvm::cl::init(false));
llvm::cl::opt<bool> Relative("relerror",
                             llvm::cl::desc("Output relative errors instead of absolute errors (experimental)."),
                             llvm::cl::init(false));
llvm::cl::opt<bool> ExactConst("exactconst",
                               llvm::cl::desc("Treat all constants as exact."),
                               llvm::cl::init(false));
llvm::cl::opt<bool> SloppyAA("sloppyaa",
                             llvm::cl::desc("Enable sloppy Alias Analysis, for when LLVM AA fails."),
                             llvm::cl::init(false));

class ErrorPropagator : public llvm::ModulePass
{
public:
  static char ID;
  ErrorPropagator() : ModulePass(ID) {}

  bool runOnModule(llvm::Module &) override;
  void getAnalysisUsage(llvm::AnalysisUsage &) const override;

protected:
  void retrieveGlobalVariablesRangeError(llvm::Module &M, RangeErrorMap &RMap);
  void checkCommandLine();

}; // end of class ErrorPropagator

} // end namespace ErrorProp

#undef DEBUG_TYPE // "errorprop"

#endif
