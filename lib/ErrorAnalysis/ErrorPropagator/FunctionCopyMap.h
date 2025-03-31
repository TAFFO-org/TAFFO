//===-- FunctionCopyMap.h - Function Clones Management ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains a class that creates and keeps track
/// of function clones, in which loops are unrolled.
///
//===----------------------------------------------------------------------===//

#ifndef ERRORPROPAGATOR_FUNCTIONCOPYMAP_H
#define ERRORPROPAGATOR_FUNCTIONCOPYMAP_H

#include <llvm/IR/Function.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <map>

#define DEBUG_TYPE "errorprop"

namespace ErrorProp
{

struct FunctionCopyCount {
  llvm::Function *Copy = nullptr;
  llvm::ValueToValueMapTy VMap;
  unsigned RecCount = 0U;
  unsigned MaxRecCount = 1U;
};

void UnrollLoops(llvm::FunctionAnalysisManager &FAM, llvm::Function &F, unsigned DefaultUnrollCount, unsigned MaxUnroll);

class FunctionCopyManager
{
public:
  FunctionCopyManager(llvm::ModuleAnalysisManager &MAM,
                      unsigned MaxRecursionCount,
                      unsigned DefaultUnrollCount,
                      unsigned MaxUnroll)
      : MAM(MAM),
        MaxRecursionCount(MaxRecursionCount),
        DefaultUnrollCount(DefaultUnrollCount),
        MaxUnroll(MaxUnroll) {}

  llvm::Function *getFunctionCopy(llvm::Function *F)
  {
    FunctionCopyCount *FCData = prepareFunctionData(F);
    assert(FCData != nullptr);

    return FCData->Copy;
  }

  unsigned getRecursionCount(llvm::Function *F)
  {
    auto FCData = FCMap.find(F);
    if (FCData == FCMap.end())
      return 0U;

    return FCData->second.RecCount;
  }

  unsigned getMaxRecursionCount(llvm::Function *F)
  {
    auto FCData = FCMap.find(F);
    if (FCData == FCMap.end())
      return MaxRecursionCount;

    return FCData->second.MaxRecCount;
  }

  void setRecursionCount(llvm::Function *F, unsigned Count)
  {
    FunctionCopyCount *FCData = prepareFunctionData(F);
    assert(FCData != nullptr);

    FCData->RecCount = Count;
  }

  unsigned incRecursionCount(llvm::Function *F)
  {
    FunctionCopyCount *FCData = prepareFunctionData(F);
    assert(FCData != nullptr);

    unsigned Old = FCData->RecCount;
    ++FCData->RecCount;

    return Old;
  }

  bool maxRecursionCountReached(llvm::Function *F)
  {
    auto FCData = FCMap.find(F);
    if (FCData == FCMap.end())
      return false;

    return FCData->second.RecCount >= FCData->second.MaxRecCount;
  }

  llvm::ValueToValueMapTy *getValueToValueMap(llvm::Function *F)
  {
    auto FCData = FCMap.find(F);
    if (FCData == FCMap.end())
      return nullptr;

    return &FCData->second.VMap;
  }

  ~FunctionCopyManager();

protected:
  typedef std::map<llvm::Function *, FunctionCopyCount> FunctionCopyMap;

  FunctionCopyMap FCMap;

  llvm::ModuleAnalysisManager &MAM;
  unsigned MaxRecursionCount;
  unsigned DefaultUnrollCount;
  unsigned MaxUnroll;

  FunctionCopyCount *prepareFunctionData(llvm::Function *F);
};

} // end namespace ErrorProp

#undef DEBUG_TYPE // "errorprop"

#endif
