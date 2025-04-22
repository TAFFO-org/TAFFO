#pragma once

#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>

namespace taffo {

/**
 * Return true if this alloca is legal for promotion.
 *
 * This is true if there are only loads, stores, and lifetime markers
 * (transitively) using this alloca. This also enforces that there is only
 * ever one layer of bitcasts or GEPs between the alloca and the lifetime
 * markers.
 */
bool isAllocaPromotable(const llvm::AllocaInst* allocaInst);

/**
 * Promote the specified list of alloca instructions into scalar
 * registers, inserting PHI nodes as appropriate.
 *
 * This function makes use of DominanceFrontier information. This function
 * does not modify the CFG of the function at all. All allocas must be from
 * the same function.
 */
void promoteMemToReg(llvm::ArrayRef<llvm::AllocaInst*> allocas,
                     llvm::DominatorTree& dominatorTree,
                     llvm::AssumptionCache* assumptionCache = nullptr);

} // namespace taffo
