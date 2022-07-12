//===-- FunctionErrorPropagator.h - Error Propagator ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Error propagator for fixed point computations in a single function.
///
//===----------------------------------------------------------------------===//

#ifndef ERRORPROPAGATOR_FUNCTIONERRORPROPAGATOR_H
#define ERRORPROPAGATOR_FUNCTIONERRORPROPAGATOR_H

#include "FunctionCopyMap.h"
#include "RangeErrorMap.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Pass.h"
#include <vector>

#define DEBUG_TYPE "errorprop"

namespace ErrorProp
{

/// Propagates errors of fixed point computations in a single function.
class FunctionErrorPropagator
{
public:
  FunctionErrorPropagator(llvm::Pass &EPPass,
                          llvm::Function &F,
                          FunctionCopyManager &FCMap,
                          mdutils::MetadataManager &MDManager,
                          bool SloppyAA)
      : EPPass(EPPass), F(F), FCMap(FCMap),
        FCopy(FCMap.getFunctionCopy(&F)), RMap(MDManager),
        CmpMap(CMPERRORMAP_NUMINITBUCKETS), MemSSA(nullptr),
        Cloned(true), SloppyAA(SloppyAA)
  {
    if (FCopy == nullptr) {
      FCopy = &F;
      Cloned = false;
    }
  }

  /// Propagate errors, cloning the function if code modifications are required.
  /// GlobRMap maps global variables and functions to their errors,
  /// and the error computed for this function's return value is stored in it;
  /// Args contains pointers to the actual parameters of a call to this function;
  /// if GenMetadata is true, computed errors are attached
  /// to each instruction as metadata.
  void computeErrorsWithCopy(RangeErrorMap &GlobRMap,
                             llvm::SmallVectorImpl<llvm::Value *> *Args = nullptr,
                             bool GenMetadata = false);

  RangeErrorMap &getRMap() { return RMap; }

protected:
  /// Compute errors instruction by instruction.
  void computeFunctionErrors(llvm::SmallVectorImpl<llvm::Value *> *ArgErrs);

  /// Compute errors for a single instruction,
  /// using the range from metadata attached to it.
  void computeInstructionErrors(llvm::Instruction &I);

  /// Compute errors for a single instruction.
  bool dispatchInstruction(llvm::Instruction &I);

  /// Compute the error on the return value of another function.
  void prepareErrorsForCall(llvm::Instruction &I);

  /// Transfer the errors computed locally to the actual parameters of the function call,
  /// but only if they are pointers.
  void applyActualParametersErrors(RangeErrorMap &GlobRMap,
                                   llvm::SmallVectorImpl<llvm::Value *> *Args);

  /// Attach error metadata to the original function.
  void attachErrorMetadata();

  /// Returns true if I may overflow, according to range data.
  bool checkOverflow(llvm::Instruction &I);

  llvm::Pass &EPPass;
  llvm::Function &F;
  FunctionCopyManager &FCMap;

  llvm::Function *FCopy;
  RangeErrorMap RMap;
  CmpErrorMap CmpMap;
  llvm::MemorySSA *MemSSA;
  bool Cloned;
  bool SloppyAA;
};

/// Schedules basic blocks of a function so that all BBs
/// that could be executed before another BB come before it in the ordering.
/// This is a sort of topological ordering that takes loops into account.
class BBScheduler
{
public:
  typedef std::vector<llvm::BasicBlock *> queue_type;
  typedef queue_type::reverse_iterator iterator;

  BBScheduler(llvm::Function &F, llvm::LoopInfo &LI)
      : Queue(), Set(), LInfo(LI)
  {
    Queue.reserve(F.size());
    enqueueChildren(&F.getEntryBlock());
  }

  bool empty() const
  {
    return Queue.empty();
  }

  iterator begin()
  {
    return Queue.rbegin();
  }

  iterator end()
  {
    return Queue.rend();
  }

protected:
  queue_type Queue;
  llvm::SmallSet<llvm::BasicBlock *, 8U> Set;
  llvm::LoopInfo &LInfo;

  /// Put BB and all of its successors in the queue.
  void enqueueChildren(llvm::BasicBlock *BB);
  /// True if Dst is an exiting or external block wrt Loop L.
  bool isExiting(llvm::BasicBlock *Dst, llvm::Loop *L) const;
};

} // end namespace ErrorProp

#undef DEBUG_TYPE // "errorprop"

#endif
