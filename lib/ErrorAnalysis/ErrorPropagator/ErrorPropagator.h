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

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/Support/CommandLine.h>

#include "FunctionCopyMap.h"
#include "RangeErrorMap.h"

#define DEBUG_TYPE "errorprop"

namespace ErrorProp
{

extern llvm::cl::opt<unsigned> DefaultUnrollCount;
extern llvm::cl::opt<unsigned> MaxUnroll;
extern llvm::cl::opt<bool> NoLoopUnroll;
extern llvm::cl::opt<unsigned> CmpErrorThreshold;
extern llvm::cl::opt<unsigned> MaxRecursionCount;
extern llvm::cl::opt<bool> StartOnly;
extern llvm::cl::opt<bool> Relative;
extern llvm::cl::opt<bool> ExactConst;
extern llvm::cl::opt<bool> SloppyAA;

class ErrorPropagator : public llvm::PassInfoMixin<ErrorPropagator>
{
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

protected:
  void retrieveGlobalVariablesRangeError(llvm::Module &M, RangeErrorMap &RMap);
  void checkCommandLine();

}; // end of class ErrorPropagator

} // end namespace ErrorProp

#undef DEBUG_TYPE // "errorprop"

#endif
