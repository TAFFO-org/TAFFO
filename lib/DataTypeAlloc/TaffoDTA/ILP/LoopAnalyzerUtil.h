//
// Created by nicola on 07/08/20.
//

#ifndef TAFFO_LOOPANALYZERUTIL_H
#define TAFFO_LOOPANALYZERUTIL_H

#include <llvm/IR/Instruction.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/PassManager.h>

#define DEBUG_TYPE "taffo-dta"

namespace tuner
{

unsigned computeFullTripCount(llvm::FunctionAnalysisManager& FAM, llvm::Instruction *instruction);
unsigned computeFullTripCount(llvm::FunctionAnalysisManager& FAM, llvm::Loop *bb);

} // namespace tuner

#undef DEBUG_TYPE // "taffo-dta"

#endif // TAFFO_LOOPANALYZERUTIL_H
