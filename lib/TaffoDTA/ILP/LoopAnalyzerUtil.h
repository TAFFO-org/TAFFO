#pragma once

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/PassManager.h>

#define DEBUG_TYPE "taffo-dta"

namespace taffo {

unsigned computeFullTripCount(llvm::FunctionAnalysisManager& FAM, llvm::Instruction* instruction);
unsigned computeFullTripCount(llvm::FunctionAnalysisManager& FAM, llvm::Loop* bb);

} // namespace taffo

#undef DEBUG_TYPE // "taffo-dta"
