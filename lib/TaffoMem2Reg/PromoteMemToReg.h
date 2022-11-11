//===- PromoteMemToReg.h - Promote Allocas to Scalars -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes an interface to promote alloca instructions to SSA
// registers, by using the SSA construction algorithm.
//
//===----------------------------------------------------------------------===//

#ifndef TAFFO_MEM2REG_PROMOTEMEMTOREG_H
#define TAFFO_MEM2REG_PROMOTEMEMTOREG_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/AssumptionCache.h"

namespace taffo {

/// Return true if this alloca is legal for promotion.
///
/// This is true if there are only loads, stores, and lifetime markers
/// (transitively) using this alloca. This also enforces that there is only
/// ever one layer of bitcasts or GEPs between the alloca and the lifetime
/// markers.
bool isAllocaPromotable(const llvm::AllocaInst *AI);

/// Promote the specified list of alloca instructions into scalar
/// registers, inserting PHI nodes as appropriate.
///
/// This function makes use of DominanceFrontier information.  This function
/// does not modify the CFG of the function at all.  All allocas must be from
/// the same function.
///
void PromoteMemToReg(llvm::ArrayRef<llvm::AllocaInst *> Allocas, llvm::DominatorTree &DT,
                     llvm::AssumptionCache *AC = nullptr);

} // End taffo namespace

#endif
