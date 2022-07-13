//
// Created by nicola on 07/08/20.
//

#ifndef TAFFO_LOOPANALYZERUTIL_H
#define TAFFO_LOOPANALYZERUTIL_H

#include "llvm/IR/Instruction.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "taffo-dta"

using namespace llvm;
class LoopAnalyzerUtil
{
public:
  static unsigned computeFullTripCount(ModulePass *tuner, Instruction *instruction);
  static unsigned computeFullTripCount(ModulePass *tuner, Loop *bb);
};

#undef DEBUG_TYPE

#endif // TAFFO_LOOPANALYZERUTIL_H
