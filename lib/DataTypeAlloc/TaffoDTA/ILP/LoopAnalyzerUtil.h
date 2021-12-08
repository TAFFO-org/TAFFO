//
// Created by nicola on 07/08/20.
//

#include "llvm/Pass.h"


#ifndef TAFFO_LOOPANALYZERUTIL_H
#define TAFFO_LOOPANALYZERUTIL_H

using namespace llvm;
class LoopAnalyzerUtil {
public:
    static unsigned computeFullTripCount(ModulePass *tuner, Instruction *instruction);
    static unsigned computeFullTripCount(ModulePass *tuner, Loop *bb);
};


#endif //TAFFO_LOOPANALYZERUTIL_H
