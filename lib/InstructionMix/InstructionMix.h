#include "llvm/IR/Instructions.h"
#include <string>
#include <map>


#ifndef INSTRUCTION_MIX_H
#define INSTRUCTION_MIX_H


class InstructionMix
{
public:
  std::map<std::string, int> stat;
  int ninstr = 0;

  void updateWithInstruction(llvm::Instruction *instr);
};

bool isFunctionInlinable(llvm::Function *fun);
int isDelimiterInstruction(llvm::Instruction *instr);
bool isSkippableInstruction(llvm::Instruction *instr);


#endif
