//===-- MemSSAUtils.h - Algorithms for MemorySSA ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Utilities for exploring MemorySSA.
///
//===----------------------------------------------------------------------===//

#ifndef TAFFO_MEMSSAUTILS_H
#define TAFFO_MEMSSAUTILS_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/MemorySSA.h"

namespace taffo
{

#define DEFAULT_RANGE_COUNT 2U

class MemSSAUtils
{
public:
  MemSSAUtils(llvm::MemorySSA &MemSSA) : MemSSA(MemSSA) {}

  llvm::SmallVectorImpl<llvm::Value *> &getDefiningValues(llvm::Instruction *i);

  static llvm::Value *getOriginPointer(llvm::MemorySSA &MemSSA, llvm::Value *Pointer);

private:
  llvm::MemorySSA &MemSSA;
  llvm::SmallSet<llvm::MemoryAccess *, DEFAULT_RANGE_COUNT> Visited;
  llvm::SmallVector<llvm::Value *, DEFAULT_RANGE_COUNT> Res;

  void findClobberingValues(llvm::Instruction *i, llvm::MemoryAccess *ma);
  void findLOEValue(llvm::Instruction *I);
  void findMemDefValue(llvm::Instruction *I, const llvm::MemoryDef *MD);
  void findMemPhiValue(llvm::Instruction *I, llvm::MemoryPhi *MPhi);
};

} // namespace taffo

#endif
