#pragma once

#include <llvm/IR/AbstractCallSite.h>

#define DEBUG_TYPE "taffo-init"

#ifdef UNITTESTS
#include <unordered_set>
bool containsUnsupportedFunctions( const llvm::Function *function, std::unordered_set<llvm::Function *> traversedFunctions);
void handleKmpcFork(const llvm::Module &m, std::vector<llvm::Instruction *> &toDelete, llvm::CallInst *curCallInstruction, const llvm::CallBase *curCall, llvm::Function *indirectFunction);
void handleIndirectCall(const llvm::Module &m, std::vector<llvm::Instruction *> &toDelete, llvm::CallInst *curCallInstruction, const llvm::CallBase *curCall, llvm::Function *indirectFunction);
#endif

namespace taffo
{

/// Check whether indirect calls are present in the given module, and patch them with dedicated trampoline calls.
/// The trampolines enable subsequent passes to better analyze the indirect calls.
void manageIndirectCalls(llvm::Module &m);

} // namespace taffo

#undef DEBUG_TYPE
