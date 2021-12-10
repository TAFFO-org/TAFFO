//
// Created by nicola on 07/08/20.
//

#ifndef TAFFO_LOOPANALYZERUTIL_H
#define TAFFO_LOOPANALYZERUTIL_H

#include "llvm/IR/Instruction.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Pass.h"

using namespace llvm;
class LoopAnalyzerUtil
{
public:
  static unsigned computeFullTripCount(ModulePass *tuner, Instruction *instruction);
  static unsigned computeFullTripCount(ModulePass *tuner, Loop *bb);
};


#endif // TAFFO_LOOPANALYZERUTIL_H
