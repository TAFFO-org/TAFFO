#include "llvm/IR/Argument.h"
#include "llvm/Support/Debug.h"

#ifndef TAFFOUTILS_DEBUGUTILS_H
#define TAFFOUTILS_DEBUGUTILS_H

#ifndef NDEBUG

#define IF_TAFFO_DEBUG \
  if (::llvm::DebugFlag && ::llvm::isCurrentDebugType(DEBUG_TYPE))

#define VALUE_ISA_WRITER(type, val) llvm::dbgs() << "isa<" #type "> " << isa<type>(val) << "\n";

#define VALUE_DUMP(val)                                                                             \
  LLVM_DEBUG(                                                                                       \
      VALUE_ISA_WRITER(Argument, val)                                                               \
          VALUE_ISA_WRITER(BasicBlock, val)                                                         \
              VALUE_ISA_WRITER(InlineAsm, val)                                                      \
                  VALUE_ISA_WRITER(MetadataAsValue, val)                                            \
                      VALUE_ISA_WRITER(User, val)                                                   \
                          VALUE_ISA_WRITER(Operator, val)                                           \
                              VALUE_ISA_WRITER(Instruction, val)                                    \
                                  VALUE_ISA_WRITER(DerivedUser, val)                                \
                                      VALUE_ISA_WRITER(Constant, val)                               \
                                          VALUE_ISA_WRITER(BlockAddress, val)                       \
                                              VALUE_ISA_WRITER(ConstantAggregate, val)              \
                                                  VALUE_ISA_WRITER(ConstantData, val)               \
                                                      VALUE_ISA_WRITER(ConstantExpr, val)           \
                                                          VALUE_ISA_WRITER(DSOLocalEquivalent, val) \
                                                              VALUE_ISA_WRITER(GlobalValue, val));


#define DUMP_STACK(val, str)            \
  {                                     \
    std::string stri;                   \
    llvm::raw_string_ostream tmp(stri); \
    tmp << *val;                        \
    if (tmp.str().find(str) == 0) {     \
      assert(0 && "DUMP STACK");        \
    }                                   \
  }


#else

#define IF_TAFFO_DEBUG if (false)

#endif

#endif // TAFFOUTILS_DEBUGUTILS_H
