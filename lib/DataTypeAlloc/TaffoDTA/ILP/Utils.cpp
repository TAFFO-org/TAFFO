#include "TaffoDTA.h"
#include "ILP/Utils.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

std::string tuner::uniqueIDForValue(Value *value)
{
  std::string buf;

  do {
    raw_string_ostream stm(buf);

    if (Instruction *inst = dyn_cast<Instruction>(value)) {
      stm << "instr_" << inst->getFunction()->getName() << "_";
    } else if (Argument *arg = dyn_cast<Argument>(value)) {
      stm << "funarg_" << arg->getParent()->getName() << "_";
    } else if (isa<Constant>(value) || isa<Operator>(value)) {
      stm << "const_";
    } else {
      llvm_unreachable("found value type unsupported by the DTA");
    }

    if (value->hasName()) {
      stm << value->getName().str();
    } else {
      value->printAsOperand(stm, false);
    }
    stm << '_';

    // add the value's pointer for good measure
    stm << std::to_string((intptr_t)value);
  } while (0);

  // sanitize
  for (auto c = buf.begin(); c != buf.end(); c++) {
    if (!(isalnum(*c) || *c == '_'))
      *c = '_';
  }

  return buf;
}

