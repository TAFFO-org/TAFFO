#include "llvm/IR/DerivedUser.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Debug.h"


#ifndef TAFFOUTILS_DEBUGUTILS_H
#define TAFFOUTILS_DEBUGUTILS_H

#ifndef NDEBUG
#define PIZZA_DEBUG(element) llvm::dbgs() << "pizza " << __FILE__ << ": " << __LINE__ << "\n" \
                                          << element << "\n";
#define VALUE_ISA_WRITER(type, val) llvm::dbgs() << "isa<" #type "> " << isa<type>(val) << "\n";

#define VALUE_DUMP(val)                                                                                   \
  LLVM_DEBUG(                                                                                             \
      VALUE_ISA_WRITER(llvm::Argument, val)                                                               \
          VALUE_ISA_WRITER(llvm::BasicBlock, val)                                                         \
              VALUE_ISA_WRITER(llvm::MetadataAsValue, val)                                                \
                  VALUE_ISA_WRITER(llvm::User, val)                                                       \
                      VALUE_ISA_WRITER(llvm::Operator, val)                                               \
                          VALUE_ISA_WRITER(llvm::Instruction, val)                                        \
                              VALUE_ISA_WRITER(llvm::DerivedUser, val)                                    \
                                  VALUE_ISA_WRITER(llvm::Constant, val)                                   \
                                      VALUE_ISA_WRITER(llvm::BlockAddress, val)                           \
                                          VALUE_ISA_WRITER(llvm::ConstantAggregate, val)                  \
                                              VALUE_ISA_WRITER(llvm::ConstantData, val)                   \
                                                  VALUE_ISA_WRITER(llvm::ConstantExpr, val)               \
                                                      VALUE_ISA_WRITER(llvm::DSOLocalEquivalent, val)     \
                                                          VALUE_ISA_WRITER(llvm::GlobalObject, val)       \
                                                              VALUE_ISA_WRITER(llvm::Function, val) \
                                                                  VALUE_ISA_WRITER(llvm::GlobalVariable, val));

#define IF_TAFFO_DEBUG \
  if (::llvm::DebugFlag && ::llvm::isCurrentDebugType(DEBUG_TYPE))

#else

#define IF_TAFFO_DEBUG if (false)

#endif

#endif // TAFFOUTILS_DEBUGUTILS_H
