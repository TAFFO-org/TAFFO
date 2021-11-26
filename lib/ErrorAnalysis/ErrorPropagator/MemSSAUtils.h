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

#ifndef ERRORPROPAGATOR_MEMSSAUTILS_H
#define ERRORPROPAGATOR_MEMSSAUTILS_H

#include "RangeErrorMap.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

namespace ErrorProp {

#define DEFAULT_RE_COUNT 2U

class MemSSAUtils {
public:
  typedef llvm::SmallVector<const RangeErrorMap::RangeError *, DEFAULT_RE_COUNT> REVector;

  MemSSAUtils(RangeErrorMap &RMap, llvm::MemorySSA &MemSSA)
    : RMap(RMap), MemSSA(MemSSA) {}

  void findMemSSAError(llvm::Instruction *I, llvm::MemoryAccess *MA);
  void findLOEError(llvm::Instruction *I);

  REVector &getRangeErrors() { return Res; }

  static llvm::Value *getOriginPointer(llvm::MemorySSA &MemSSA, llvm::Value *Pointer);

private:
  RangeErrorMap &RMap;
  llvm::MemorySSA &MemSSA;
  llvm::SmallSet<llvm::MemoryAccess *, DEFAULT_RE_COUNT> Visited;
  REVector Res;

  void findMemDefError(llvm::Instruction *I, const llvm::MemoryDef *MD);
  void findMemPhiError(llvm::Instruction *I, llvm::MemoryPhi *MPhi);
};

} // end of namespace ErrorProp

#endif
